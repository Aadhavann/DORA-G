"""
Script 2: Build FAISS retrieval index from CodeSearchNet.
"""

import sys
sys.path.append("..")

import hydra
from omegaconf import DictConfig
from pathlib import Path
from datasets import load_from_disk
from src.retrieval.retriever import CodeRetriever
from src.data.preprocessing import CodeSearchNetPreprocessor
from src.utils.reproducibility import set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="rag")
def main(config: DictConfig):
    """Build retrieval index from CodeSearchNet."""
    print("="*80)
    print("STEP 2: Building Retrieval Index")
    print("="*80)

    # Set seed
    set_seed(config.reproducibility.seed)

    # Load CodeSearchNet
    data_dir = Path(config.paths.data_dir)
    codesearchnet_path = data_dir / "codesearchnet"

    if not codesearchnet_path.exists():
        print(f"Error: CodeSearchNet not found at {codesearchnet_path}")
        print("Please run 01_prepare_data.py first")
        return

    print(f"Loading CodeSearchNet from {codesearchnet_path}...")
    dataset = load_from_disk(str(codesearchnet_path))

    # Preprocess for retrieval
    print("Preprocessing code for retrieval...")
    preprocessor = CodeSearchNetPreprocessor()

    processed_dataset = dataset.map(
        preprocessor.preprocess_for_retrieval,
        desc="Preprocessing",
    )

    # Initialize retriever
    print("Initializing retriever...")
    retriever = CodeRetriever(config)

    # Build index
    print("Building FAISS index (this may take a while)...")
    retriever.build_index_from_dataset(
        dataset=processed_dataset,
        text_field="text"
    )

    # Save index
    index_dir = Path(config.rag.index.index_path)
    index_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving index to {index_dir}...")
    retriever.save(str(index_dir))

    # Test retrieval
    print("\n--- Testing Retrieval ---")
    test_query = "function to read a CSV file"
    results = retriever.retrieve(test_query, top_k=3)

    print(f"Query: {test_query}")
    print("\nTop 3 results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Function: {result.get('func_name', 'N/A')}")
        print(f"   Code preview: {result.get('code', '')[:100]}...")

    print("\n" + "="*80)
    print("Index building complete!")
    print("="*80)


if __name__ == "__main__":
    main()
