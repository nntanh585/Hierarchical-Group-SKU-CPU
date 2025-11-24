import pandas as pd
import numpy as np
from light_embed import TextEmbedding
import hdbscan
import faiss
from typing import List, Dict, Tuple, Optional

from rakuten_processor_hierarchical import (
    _transform_data_from_json,
    _get_master_text,
    _get_variant_text,
)


class HierarchicalHdbscanAssigner:
    """
    A hierarchical product clustering and assignment engine designed for:
      - Master-level clustering using HDBSCAN
      - Variant-level clustering using local cosine similarity
      - Real-time assignment for new incoming products
      - Master retrieval acceleration using FAISS

    This class solves the SKU fragmentation problem in marketplaces by
    automatically grouping products into stable Master/Variant clusters.
    """

    # Initialization
    def __init__(
        self,
        embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        min_cluster_size: int = 2,
        master_similarity_threshold: float = 0.85,
        variant_similarity_threshold: float = 0.95,
    ):
        print(f"Initializing embedding model: {embed_model_name}...")
        self.embed_model_name = embed_model_name
        self.model: Optional[TextEmbedding] = None
        self.min_cluster_size = min_cluster_size
        self.master_threshold = master_similarity_threshold
        self.variant_threshold = variant_similarity_threshold

        # Runtime state containers
        self.master_index: Optional[faiss.IndexFlatIP] = None
        self.df_master: Optional[pd.DataFrame] = None
        self.variant_store: Dict[int, List[Dict]] = {}  # Per-master variant clusters
        self.next_master_id: int = 0

        self._load_model()

    # Model Loading
    def _load_model(self):
        """Load the SBERT encoder used for vectorization."""
        print(f"Loading SBERT model: {self.embed_model_name} ...")
        self.model = TextEmbedding(self.embed_model_name)
        print("Model loaded.")

    # Initial Clustering Pipeline
    def build_initial_clusters(self, json_file: str):
        """
        Perform a full offline bootstrap process:
          1. Load dataset from JSON
          2. Vectorize master-level texts
          3. Run HDBSCAN to form master clusters
          4. Build FAISS index for master retrieval
          5. For each master group, encode and cluster variants locally
        """
        print(f"--- Bootstrapping from {json_file} ---")
        # Step 1: Load + Transform
        df = _transform_data_from_json(json_file)

        # Step 2: Encode Master Embeddings
        print("Encoding master embeddings...")
        master_texts = df.apply(_get_master_text, axis=1).tolist()
        master_vecs = (
            self.model.encode(
                master_texts,
                normalize_embeddings=True
            ).astype("float32")
        )

        # Step 3: HDBSCAN Clustering
        print("Running HDBSCAN for master clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(master_vecs)

        # Assign outliers new incremental master IDs
        max_label = max(labels)
        if max_label == -1:
            max_label = 0

        next_id = max_label + 1
        final_master_ids = []

        for label in labels:
            if label != -1:
                final_master_ids.append(label)
            else:
                final_master_ids.append(next_id)
                next_id += 1

        df["master_id"] = final_master_ids
        self.next_master_id = next_id

        print(f"Total Master Groups Identified: {self.next_master_id}")

        # Step 4: Build FAISS index
        print("Building FAISS index for masters...")
        self.master_index = faiss.IndexFlatIP(master_vecs.shape[1])
        self.master_index.add(master_vecs)

        # Step 5: Variant Clustering (Per-group)
        print("Clustering variants group-by-group...")

        variant_id_map = {}
        grouped = df.groupby("master_id")

        for m_id, group_df in grouped:
            self.variant_store[m_id] = []

            # Encode variant-level text only for that group
            variant_texts = group_df.apply(_get_variant_text, axis=1).tolist()

            variant_vecs = (
                self.model.encode(
                    variant_texts,
                    normalize_embeddings=True
                ).astype("float32")
            )

            # Local variant similarity clustering
            for i, (row_idx, row) in enumerate(group_df.iterrows()):
                v_text = variant_texts[i]
                v_vec = variant_vecs[i]

                current_variants = self.variant_store[m_id]
                assigned_id = -1
                best_sim = -1.0

                for v in current_variants:
                    sim = float(np.dot(v_vec, v["vector"]))
                    if sim > best_sim:
                        best_sim = sim
                        if sim >= self.variant_threshold:
                            assigned_id = v["variant_id"]
                            break

                if assigned_id == -1:
                    assigned_id = len(current_variants)
                    self.variant_store[m_id].append(
                        {
                            "variant_id": assigned_id,
                            "vector": v_vec,
                            "specs": v_text,
                            "original_sku": row.get("seller_sku"),
                        }
                    )

                variant_id_map[row_idx] = assigned_id

        df["variant_id"] = df.index.map(variant_id_map)
        self.df_master = df

        print("--- Bootstrap Complete ---")

    # Real-time Assignment for New Products
    def assign_new_product(
        self,
        product_dict: Dict,
        master_score: Optional[float] = None,
        variant_score: Optional[float] = None,
    ) -> Tuple[int, int]:
        """
        Assign a new product to an existing master & variant cluster.
        If similarity thresholds are not met, new clusters will be created.
        """

        if self.master_index is None:
            raise RuntimeError("Run build_initial_clusters() before assigning new products.")

        # Allow thresholds to be overridden per-request
        if master_score is not None:
            self.master_threshold = master_score
        if variant_score is not None:
            self.variant_threshold = variant_score

        row = pd.Series(product_dict)

        # Step 1: Master Assignment
        m_text = _get_master_text(row)
        m_vec = self.model.encode([m_text], normalize_embeddings=True).astype("float32")

        D, I = self.master_index.search(m_vec, 1)
        sim = float(D[0][0])
        nearest_idx = int(I[0][0])

        if sim >= self.master_threshold:
            master_id = int(self.df_master.iloc[nearest_idx]["master_id"])
        else:
            master_id = self.next_master_id
            self.next_master_id += 1
            self.variant_store[master_id] = []

        # Step 2: Variant Assignment
        v_text = _get_variant_text(row)
        v_vec = self.model.encode([v_text], normalize_embeddings=True).astype("float32")[0]

        candidates = self.variant_store[master_id]
        variant_id = -1
        best_sim = -1.0

        for v in candidates:
            sim = float(np.dot(v_vec, v["vector"]))
            if sim > best_sim:
                best_sim = sim
                if sim >= self.variant_threshold:
                    variant_id = v["variant_id"]
                    break

        if variant_id == -1:
            variant_id = len(candidates)
            self.variant_store[master_id].append(
                {
                    "variant_id": variant_id,
                    "vector": v_vec,
                    "specs": v_text,
                    "original_sku": row.get("seller_sku"),
                }
            )

        # Step 3: Persist new entry
        self.master_index.add(m_vec)  # Append master vector

        new_row = row.copy()
        new_row["master_id"] = master_id
        new_row["variant_id"] = variant_id
        self.df_master = pd.concat([self.df_master, pd.DataFrame([new_row])], ignore_index=True)

        return master_id, variant_id
