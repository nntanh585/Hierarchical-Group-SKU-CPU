import re
import json
import pandas as pd

# Helper functions
def _read_json(json_file: str) -> dict:
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def _clean_text(text: str) -> str:
    """Clean up, normalize to lowercase and remove special characters"""
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text) # Remove HTML tags
    text = re.sub(r"\[.*?\]", " ", text) # Remove text in square brackets
    text = re.sub(r"[^\w\s]", " ", text) # Remove punctuation
    text = re.sub(r"\s+", " ", text).strip() # Remove extra whitespace
    return text

def _extract_adaptive_attributes(attributes_list: list) -> dict:
    """
    Extract all adaptive key-value pairs.
    """
    result_dict = {}
    if not attributes_list:
        return result_dict

    for attr in attributes_list:
        key = None
        value = None

        # Case 1: Structure 'required_attributes'
        if 'name' in attr and 'value' in attr:
            key_raw = attr.get('name')
            value_list = attr.get('value', [])
            if key_raw and value_list and value_list[0]:
                key = _clean_text(key_raw)
                value = _clean_text(value_list[0])
                unit = attr.get('unit')
                if unit:
                    value = f"{value} {unit}"

        # Case 2: Structure 'attribute_simples'
        elif 'attributes' in attr and 'attribute_values' in attr:
            key_raw = attr.get('attributes', {}).get('name')
            value_raw = attr.get('attribute_values', {}).get('name')
            if key_raw and value_raw:
                key = _clean_text(key_raw)
                value = _clean_text(value_raw)

        if key and value:
            result_dict[key] = value

    return result_dict

def _transform_data_from_json(json_file: str) -> pd.DataFrame:
    """
    Read JSON file and flatten the data.
    """
    full_json_data = _read_json(json_file)
    products_list = full_json_data.get('data', []) 

    result_list = []

    for product_data in products_list:
        # 1. Parent information
        parent_id = product_data.get('id')
        parent_name = _clean_text(product_data.get('name'))
        parent_category = _clean_text(product_data.get('category', {}).get('name_en'))
        parent_short_desc = _clean_text(product_data.get('short_description'))
        
        # Safely handle jan_info
        jan_info = product_data.get('jan_info', {})
        jan_code = ""
        if isinstance(jan_info, dict):
            jan_code = str(jan_info.get('reason_no_code') or jan_info.get('code') or "")

        # 2. Loop through Children (SKUs)
        skus_data = product_data.get('product_skus', {}).get('data', [])
        if skus_data:
            for sku in skus_data:
                req_attrs_list = sku.get('required_attributes', [])
                simple_attrs_list = sku.get('attribute_simples', {}).get('data', [])

                attributes_dict = _extract_adaptive_attributes(req_attrs_list)
                attributes_dict.update(_extract_adaptive_attributes(simple_attrs_list))

                product_info = {
                    "product_id": parent_id,
                    "product_name": parent_name,
                    "category_name": parent_category,
                    "short_description": parent_short_desc,
                    "seller_sku": _clean_text(sku.get("seller_sku")),
                    "attributes": attributes_dict,
                    "jan_infor": jan_code
                }
                result_list.append(product_info)

    return pd.DataFrame(result_list) if result_list else pd.DataFrame()

# Text generation functions
def _get_master_text(product_row: pd.Series) -> str:
    """
    Create text string for MASTER clustering.
    Goal: Group variants (Color/Size) into the same group.
    Strategy: Only take common information (Name, Brand, Model, Category, JAN).
              EXCLUDE SKU and variant attributes.
    """
    name = product_row.get('product_name', '')
    category = product_row.get('category_name', '')
    short_description = product_row.get('short_description', '')
    jan_infor = product_row.get('jan_infor', '')
    sku = product_row.get('seller_sku', '')

    # Get attributes to find Brand/Model, but not all
    attributes_dict = product_row.get('attributes', {})
    
    # Try to find Brand and Model in the messy attributes
    brand = ""
    model = ""
    
    # Common keywords to identify Brand/Model (cleaned text)
    for k, v in attributes_dict.items():
        if 'brand' in k or 'ブランド' in k: # Brand name
            brand = v
        if 'model' in k or '型番' in k: # Model number
            model = v

    # Prompt structure for Master (No SKU, no specific Color/Size)
    text = f"Product: {name}\nSKU: {sku}\nBrand: {brand}\nModel: {model}\nCategory: {category}\nJAN: {jan_infor}\nDescription: {short_description}"
    return text

def _get_variant_text(product_row: pd.Series) -> str:
    """
    Create text string for VARIANT clustering.
    Goal: Differentiate between SKUs within the same Master group.
    Strategy: Focus on SKU and ALL detailed attributes.
    """
    # Still need the name to maintain context (e.g., distinguish Shirt vs Pants if Master grouping is incorrect)
    # But the main weight will be on Details and SKU
    name = product_row.get('product_name', '')
    attributes_dict = product_row.get('attributes', {})

    # Remove unwanted keys (as requested by user)
    remove_keys = {'総個数', '総重量', '総容量'}
    filtered_attrs = {k: v for k, v in attributes_dict.items() if k not in remove_keys}
    
    # Create attribute details string
    details = " | ".join([f"{k}: {v}" for k, v in filtered_attrs.items()])

    # Prompt structure for Variant (SKU and Details are most important)
    text = f"Details: {details}\nContext: {name}"
    return text