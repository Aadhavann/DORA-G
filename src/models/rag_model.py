"""
RAG-augmented model wrapper for retrieval-augmented generation.
"""

from typing import Optional, List, Dict, Any
from omegaconf import DictConfig


class RAGModel:
    """
    Wrapper that augments any base model with RAG capabilities.
    This can wrap BaseCodeModel, LoRAModel, or DoRAModel.
    """

    def __init__(self, base_model, retriever, config: DictConfig):
        """
        Initialize RAG-augmented model.

        Args:
            base_model: The underlying model (BaseCodeModel, LoRAModel, or DoRAModel)
            retriever: Retriever instance for fetching relevant code
            config: Hydra configuration with RAG settings
        """
        self.base_model = base_model
        self.retriever = retriever
        self.config = config
        self.rag_config = config.rag

        # Prompt template configuration
        self.separator = self.rag_config.prompt.separator
        self.code_format = self.rag_config.prompt.code_format
        self.instruction = self.rag_config.prompt.instruction

    def _format_retrieved_code(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved code snippets for injection into prompt.

        Args:
            retrieved_docs: List of retrieved documents with 'code' and metadata

        Returns:
            Formatted string with retrieved code examples
        """
        if not retrieved_docs:
            return ""

        formatted_code = self.instruction
        for i, doc in enumerate(retrieved_docs, 1):
            code = doc.get("code", doc.get("text", ""))
            # Format code with language-specific markers
            formatted_code += self.code_format.format(code=code)

        return formatted_code

    def _augment_prompt(self, prompt: str, query: Optional[str] = None) -> str:
        """
        Augment prompt with retrieved code examples.

        Args:
            prompt: Original user prompt/instruction
            query: Optional custom query for retrieval (defaults to prompt)

        Returns:
            Augmented prompt with retrieved code context
        """
        # Use prompt as query if not specified
        retrieval_query = query if query is not None else prompt

        # Retrieve relevant code snippets
        retrieved_docs = self.retriever.retrieve(
            query=retrieval_query,
            top_k=self.rag_config.retrieval.top_k,
        )

        # Filter by similarity threshold if configured
        min_similarity = self.rag_config.retrieval.get("min_similarity", 0.0)
        if min_similarity > 0:
            retrieved_docs = [
                doc for doc in retrieved_docs
                if doc.get("score", 1.0) >= min_similarity
            ]

        if not retrieved_docs:
            # No relevant retrieval results, return original prompt
            return prompt

        # Format retrieved code
        retrieved_context = self._format_retrieved_code(retrieved_docs)

        # Inject into prompt
        augmented_prompt = f"{retrieved_context}{self.separator}{prompt}"

        return augmented_prompt

    def generate(
        self,
        prompt: str,
        retrieval_query: Optional[str] = None,
        use_rag: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """
        Generate code with optional RAG augmentation.

        Args:
            prompt: Input prompt
            retrieval_query: Optional custom query for retrieval
            use_rag: Whether to use RAG (can disable for ablation)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters

        Returns:
            Generated code
        """
        # Augment prompt with retrieval if enabled
        if use_rag and self.rag_config.enabled:
            augmented_prompt = self._augment_prompt(prompt, retrieval_query)
        else:
            augmented_prompt = prompt

        # Generate using base model
        generated = self.base_model.generate(
            prompt=augmented_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs
        )

        return generated

    def get_retrieval_context(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get retrieval context without generation (for analysis).

        Args:
            query: Retrieval query
            top_k: Number of results to retrieve

        Returns:
            List of retrieved documents with metadata
        """
        k = top_k if top_k is not None else self.rag_config.retrieval.top_k
        return self.retriever.retrieve(query=query, top_k=k)

    # Delegate attribute access to base model
    def __getattr__(self, name):
        """Delegate unknown attributes to base model."""
        return getattr(self.base_model, name)
