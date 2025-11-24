import psutil, os, gc, time
from group_hierarchical import HierarchicalHdbscanAssigner
from rakuten_processor_hierarchical import (
    _transform_data_from_json
)

def measure_memory(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        gc.collect()
        mem_before = process.memory_info().rss / (1024 ** 2)
        start = time.time()

        result = func(*args, **kwargs)

        gc.collect()
        mem_after = process.memory_info().rss / (1024 ** 2)
        end = time.time()

        print(f"\n[Memory Profiling] {func.__name__}")
        print(f"RAM before: {mem_before:.2f} MB")
        print(f"RAM after:  {mem_after:.2f} MB")
        print(f"Δ RAM used: {mem_after - mem_before:.2f} MB")
        print(f"Execution time: {end - start:.2f}s")
        print("-" * 50)
        return result
    return wrapper

HierarchicalHdbscanAssigner._load_model = measure_memory(HierarchicalHdbscanAssigner._load_model)
HierarchicalHdbscanAssigner.build_initial_clusters = measure_memory(HierarchicalHdbscanAssigner.build_initial_clusters)

assigner = HierarchicalHdbscanAssigner(
    embed_model_name="sentence-transformers/all-MiniLM-L6-v2",
    min_cluster_size=2,       
    master_similarity_threshold=0.8, 
    variant_similarity_threshold=0.999
)

json_filename = "./data.json"
assigner.build_initial_clusters(json_filename)

df = _transform_data_from_json("./sample.json")
variant_score = 0.993
for index, row in df.iterrows():
    master_id, variant_id = assigner.assign_new_product(
        product_dict=row,
        variant_score=variant_score
    )
    print(master_id, variant_id)
    
