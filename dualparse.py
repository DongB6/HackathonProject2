import os
import json
import requests
import zipfile
from io import BytesIO
from lxml import etree

DATA_DIR = "data"
OUTPUT_DIR = "output"

# URLs
HEALTH_TOPICS_ZIP_URL = "https://medlineplus.gov/xml/mplus_topics_compressed_2025-06-14.zip"
DEFINITIONS_XML_URL = "https://medlineplus.gov/xml/generalhealthdefinitions.xml"

# Local file paths
HEALTH_TOPICS_ZIP_PATH = os.path.join(DATA_DIR, "mplus_topics_compressed_2025-06-14.zip")
HEALTH_TOPICS_XML_PATH = os.path.join(DATA_DIR, "mplus_topics_2025-06-14.xml")
DEFINITIONS_XML_PATH = os.path.join(DATA_DIR, "generalhealthdefinitions.xml")

# Output files
COMBINED_JSON_PATH = os.path.join(OUTPUT_DIR, "medical_terms_combined.json")
SEARCH_INDEX_PATH = os.path.join(OUTPUT_DIR, "search_index.json")
METADATA_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File already exists: {dest_path}")
        return
    print(f"Downloading {url} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(response.content)
    print(f"Saved to {dest_path}")

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def parse_health_topics(xml_path):
    print(f"Parsing Health Topics XML: {xml_path}")
    tree = etree.parse(xml_path)
    root = tree.getroot()

    topics = []

    # Each 'healthTopic' element = medical term
    for topic in root.findall(".//healthTopic"):
        title = topic.findtext("title", default="").strip()
        description = topic.findtext("definition/simple", default="").strip()

        symptoms = topic.findtext("healthTopicDetails/symptoms", default="").strip()
        causes = topic.findtext("healthTopicDetails/causes", default="").strip()
        prevention = topic.findtext("healthTopicDetails/prevention", default="").strip()
        treatment = topic.findtext("healthTopicDetails/treatment", default="").strip()

        # Normalize empty strings
        also_called = []
        for alt in topic.findall("alsoCalled/term"):
            alt_name = alt.text.strip()
            if alt_name:
                also_called.append(alt_name)

        # URL 
        url = topic.findtext("url", default="").strip()

        topics.append({
            "title": title,
            "type": "health_topic",
            "description": description,
            "symptoms": symptoms,
            "causes": causes,
            "prevention": prevention,
            "treatment": treatment,
            "also_called": also_called,
            "url": url,
            "source": "MedlinePlus Health Topics"
        })

    print(f"Parsed {len(topics)} health topics")
    return topics

def parse_definitions(xml_path):
    print(f"Parsing Definitions XML: {xml_path}")
    tree = etree.parse(xml_path)
    root = tree.getroot()

    definitions = []

    for group in root.findall(".//term-group"):
        term_elem = group.find("term")
        def_elem = group.find("definition")

        title = term_elem.text.strip() if term_elem is not None and term_elem.text else ""
        if title.startswith(">"):
            title = title[1:].strip()

        definition = def_elem.text.strip() if def_elem is not None and def_elem.text else ""
        if definition.startswith(">"):
            definition = definition[1:].strip()

        reference = group.get("reference", "")
        reference_url = group.get("reference-url", "")

        definitions.append({
            "title": title,
            "type": "definition",
            "description": definition,
            "symptoms": "",
            "causes": "",
            "prevention": "",
            "treatment": "",
            "also_called": [],
            "url": reference_url,
            "source": reference or "MedlinePlus Definitions"
        })

    print(f"Parsed {len(definitions)} definitions")
    return definitions


def combine_datasets(health_topics, definitions):
    print("Combining datasets, removing duplicates...")
    combined = {}
    # Use title as key, normalize to lowercase for matching
    for item in health_topics:
        combined[item['title'].lower()] = item

    # Add definitions only if title not already present 
    # (health topics prioritized)
    for item in definitions:
        key = item['title'].lower()
        if key not in combined:
            combined[key] = item

    combined_list = list(combined.values())
    print(f"Combined dataset size: {len(combined_list)}")
    return combined_list

def build_search_index(terms):
    print("Building search index...")
    index = {}
    for term in terms:
        title = term['title'].lower()
        index[title] = term

        # Index also_called terms for lookup
        for alt in term.get('also_called', []):
            alt_key = alt.lower()
            if alt_key not in index:
                index[alt_key] = term

    print(f"Search index size: {len(index)}")
    return index

def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved JSON to {path}")

def main():
    ensure_dirs()

    # Download files if missing
    download_file(HEALTH_TOPICS_ZIP_URL, HEALTH_TOPICS_ZIP_PATH)
    download_file(DEFINITIONS_XML_URL, DEFINITIONS_XML_PATH)

    # Extract health topics XML from zip
    if not os.path.exists(HEALTH_TOPICS_XML_PATH):
        extract_zip(HEALTH_TOPICS_ZIP_PATH, DATA_DIR)

    # Parse datasets
    health_topics = parse_health_topics(HEALTH_TOPICS_XML_PATH)

    # Try parsing definitions, if file exists
    if os.path.exists(DEFINITIONS_XML_PATH):
        definitions = parse_definitions(DEFINITIONS_XML_PATH)
    else:
        print(f"Warning: Definitions file not found at {DEFINITIONS_XML_PATH}. Skipping definitions dataset.")
        definitions = []

    # Combine and deduplicate
    combined_terms = combine_datasets(health_topics, definitions)

    # Build search index
    search_index = build_search_index(combined_terms)

    # Save outputs
    save_json({
        "terms": combined_terms,
        "total_count": len(combined_terms),
        "health_topics_count": len(health_topics),
        "definitions_count": len(definitions)
    }, COMBINED_JSON_PATH)

    save_json(search_index, SEARCH_INDEX_PATH)

    save_json({
        "health_topics_file": HEALTH_TOPICS_XML_PATH,
        "definitions_file": DEFINITIONS_XML_PATH if os.path.exists(DEFINITIONS_XML_PATH) else None,
        "total_terms": len(combined_terms)
    }, METADATA_PATH)

    print("All done!")

if __name__ == "__main__":
    main()
