import numpy as np
from PIL import Image
import requests

# We would need a CLIP model and tokenizer/processor from HuggingFace.
# For example:
# from transformers import CLIPProcessor, CLIPModel


class MediaLoader:
    """Handles loading of images from URLs or disk, and other media types if needed."""

    def load_image(self, source: str) -> Image.Image:
        """Load an image from a URL or local path into a PIL Image."""
        if source.startswith("http://") or source.startswith("https://"):
            response = requests.get(source)
            response.raise_for_status()
            data = response.content
        else:
            # Assume it's a local file path
            with open(source, "rb") as f:
                data = f.read()
        # Open the image data with PIL
        image = Image.open(bytearray(data))
        return image.convert("RGB")  # ensure 3-channel RGB

    def load_text(self, text: str) -> str:
        """For symmetry, just returns the text (could add file reading if needed)."""
        # If needed, handle file pointers, etc. For now, assume text is raw.
        return text


class Embedder:
    """Abstract base embedder class."""

    def embed_text(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def embed_image(self, image: Image.Image) -> np.ndarray:
        raise NotImplementedError


class CLIPEmbedder(Embedder):
    """Embedder that uses a CLIP model for text and images."""

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        # Load the CLIP model and processor from HuggingFace
        # (Note: In an offline environment, this would require cached models)
        # self.model = CLIPModel.from_pretrained(model_name)
        # self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = None  # Placeholder for actual model
        self.processor = None  # Placeholder for actual processor
        # For this conceptual code, we'll assume model and processor are set up.

    def embed_text(self, text: str) -> np.ndarray:
        # Prepare the text and get embeddings from the model
        # inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        # with torch.no_grad():
        #     outputs = self.model.get_text_features(**inputs)
        # vector = outputs[0].numpy()
        # For conceptual purposes, we simulate with a random vector:
        vector = np.random.rand(512).astype("float32")
        # Normalize the vector (CLIP outputs are often already normalized, but just in case)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def embed_image(self, image: Image.Image) -> np.ndarray:
        # Prepare the image and get embeddings from the model
        # inputs = self.processor(images=image, return_tensors="pt")
        # with torch.no_grad():
        #     outputs = self.model.get_image_features(**inputs)
        # vector = outputs[0].numpy()
        # Simulate with random vector for conceptual demonstration:
        vector = np.random.rand(512).astype("float32")
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector


class DocumentEncoder:
    """Encodes documents (with possible multimodal fields) into embeddings using an Embedder."""

    def __init__(self, embedder: Embedder, media_loader: MediaLoader):
        self.embedder = embedder
        self.media_loader = media_loader

    def encode(
        self, document: dict, mappings: dict = None, tensor_fields: list = None
    ) -> dict:
        """
        Encode the specified tensor_fields of the document.
        Returns a dict of field name to embedding vector (np.ndarray).
        """
        if mappings is None:
            mappings = {}
        if tensor_fields is None:
            # If not specified, assume all fields are tensor fields
            tensor_fields = list(document.keys())

        encoded_vectors = {}
        for field in tensor_fields:
            if (
                field in mappings
                and mappings[field].get("type") == "multimodal_combination"
            ):
                # Multimodal combination field: combine child fields with weights
                weights = mappings[field].get("weights", {})
                combined_vec = None
                total_weight = 0.0
                for sub_field, weight in weights.items():
                    if sub_field not in document:
                        continue  # skip if sub_field missing in doc
                    content = document[sub_field]
                    # Determine modality by type of content
                    if isinstance(content, str):
                        # Heuristic: treat as image if it's a URL or file path ending in image extension
                        if (
                            content.startswith("http")
                            or content.startswith("/")
                            or content.lower().endswith(
                                (".png", ".jpg", ".jpeg")
                            )
                        ):
                            # likely an image pointer
                            img = self.media_loader.load_image(content)
                            vec = self.embedder.embed_image(img)
                        else:
                            # treat as text
                            text = self.media_loader.load_text(content)
                            vec = self.embedder.embed_text(text)
                    else:
                        # If content is already an image object or array, embed directly
                        if isinstance(content, Image.Image):
                            vec = self.embedder.embed_image(content)
                        else:
                            # For other types (e.g., if someone precomputed a vector), skip or integrate accordingly
                            continue
                    # Accumulate weighted sum
                    if combined_vec is None:
                        combined_vec = weight * vec
                    else:
                        combined_vec += weight * vec
                    total_weight += weight
                if combined_vec is None:
                    continue  # nothing to combine
                # (Optionally normalize combined vector or divide by total_weight if weights not meant to be absolute)
                # Here, we assume weights are relative importance, not necessarily summing to 1.
                # We could normalize by total_weight to keep scale consistent:
                if total_weight != 0:
                    combined_vec = combined_vec / total_weight
                # Normalize the combined vector as well
                norm = np.linalg.norm(combined_vec)
                if norm > 0:
                    combined_vec = combined_vec / norm
                encoded_vectors[field] = combined_vec
            else:
                # Single field embedding (text or image)
                if field not in document:
                    continue
                content = document[field]
                if isinstance(content, str):
                    # Decide modality by content
                    if (
                        content.startswith("http")
                        or content.startswith("/")
                        or content.lower().endswith((".png", ".jpg", ".jpeg"))
                    ):
                        img = self.media_loader.load_image(content)
                        vec = self.embedder.embed_image(img)
                    else:
                        text = self.media_loader.load_text(content)
                        vec = self.embedder.embed_text(text)
                elif isinstance(content, Image.Image):
                    vec = self.embedder.embed_image(content)
                else:
                    # If it's e.g. bytes for an image, we can load it via PIL
                    try:
                        img = Image.open(content)
                        vec = self.embedder.embed_image(img)
                    except Exception:
                        # Fallback: if content is some other type, ignore or extend logic as needed
                        continue
                encoded_vectors[field] = vec
        return encoded_vectors


class VectorIndex:
    """Simple vector index for storing embeddings and performing similarity search."""

    def __init__(self, similarity: str = "cosine"):
        self.vectors = []  # list of np.ndarray
        self.docs = []  # parallel list of documents or IDs
        self.similarity = similarity

    def add(self, doc, vector: np.ndarray):
        """Add a document (or doc ID and content) with its vector to the index."""
        # Store a copy of the vector (normalized already by embedder) and the doc reference
        self.vectors.append(vector.astype("float32"))
        self.docs.append(doc)

    def search(self, query_vector: np.ndarray, top_k: int = 5):
        """Return the top_k most similar documents to the query_vector."""
        if self.similarity == "cosine":
            # Ensure query_vector is normalized
            q = query_vector
            norm = np.linalg.norm(q)
            if norm > 0:
                q = q / norm
            # Compute cosine similarity as dot product (since vectors are normalized)
            scores = []
            for v in self.vectors:
                # Normalize stored vector as well (should already be normalized)
                vv = v
                norm_v = np.linalg.norm(vv)
                if norm_v > 0:
                    vv = vv / norm_v
                sim = float(np.dot(q, vv))
                scores.append(sim)
            # Get top_k indices
            if len(scores) == 0:
                return []
            scores = np.array(scores)
            top_idx = np.argpartition(scores, -top_k)[-top_k:]
            # Sort those indices by score (descending for similarity)
            top_idx = top_idx[np.argsort(-scores[top_idx])]
            results = []
            for idx in top_idx:
                results.append((self.docs[idx], scores[idx]))
            return results
        else:
            # Implement other similarity metrics if needed (euclidean, etc.)
            raise NotImplementedError(
                "Only cosine similarity is implemented in this example."
            )


class SearchEngine:
    """High-level search engine that ties all components together."""

    def __init__(self, embedder: Embedder = None):
        self.media_loader = MediaLoader()
        # If no embedder provided, use a default CLIPEmbedder
        self.embedder = embedder if embedder is not None else CLIPEmbedder()
        self.doc_encoder = DocumentEncoder(self.embedder, self.media_loader)
        self.index = VectorIndex()
        self._doc_id_counter = 0  # to assign IDs if documents don't have one
        # Store original docs by ID if needed (for returning results)
        self._documents = {}

    def add_documents(
        self, documents: list, mappings: dict = None, tensor_fields: list = None
    ):
        """Add multiple documents to the index with given mappings and tensor fields."""
        for doc in documents:
            # Assign an ID if the document has none (we assume each doc is a dict)
            doc_id = None
            if "_id" in doc:
                doc_id = doc["_id"]
            else:
                doc_id = f"doc{self._doc_id_counter}"
                self._doc_id_counter += 1
            # Encode document to get its vectors
            encoded = self.doc_encoder.encode(
                doc, mappings=mappings, tensor_fields=tensor_fields
            )
            # If there are combined/multiple vectors, we might choose one to index or index all.
            # For simplicity, if multiple vectors exist, index them separately associated with the same doc.
            for field, vector in encoded.items():
                # We can store (doc_id, field) as the reference, or store the entire doc
                reference = {"_id": doc_id, "_source": doc, "_field": field}
                self.index.add(reference, vector)
            # Keep original document
            self._documents[doc_id] = doc
        return True  # could return some status or IDs

    def search(self, query, top_k: int = 5):
        """Search the index with a text query, image URL, or weighted query dict."""
        # Determine if query is weighted (dict) or a single query
        query_vector = None
        if isinstance(query, dict):
            # Weighted query: keys are sub-queries (text or image), values are weights
            combined_q = None
            total_weight = 0.0
            for sub_q, weight in query.items():
                # For each sub-query, get its embedding vector
                vec = None
                if isinstance(sub_q, str):
                    # Check if this sub-query is an image URL
                    if (
                        sub_q.startswith("http")
                        or sub_q.startswith("/")
                        or sub_q.lower().endswith((".png", ".jpg", ".jpeg"))
                    ):
                        img = self.media_loader.load_image(sub_q)
                        vec = self.embedder.embed_image(img)
                    else:
                        text = self.media_loader.load_text(sub_q)
                        vec = self.embedder.embed_text(text)
                elif isinstance(sub_q, Image.Image):
                    vec = self.embedder.embed_image(sub_q)
                else:
                    # Unsupported query type
                    continue
                # Accumulate weighted vectors
                if vec is not None:
                    if combined_q is None:
                        combined_q = weight * vec
                    else:
                        combined_q += weight * vec
                    total_weight += weight
            if combined_q is None:
                return []  # no valid subqueries
            if total_weight != 0:
                combined_q = combined_q / total_weight
            query_vector = combined_q
        else:
            # Single query (text or image)
            if isinstance(query, str):
                if (
                    query.startswith("http")
                    or query.startswith("/")
                    or query.lower().endswith((".png", ".jpg", ".jpeg"))
                ):
                    img = self.media_loader.load_image(query)
                    query_vector = self.embedder.embed_image(img)
                else:
                    text = self.media_loader.load_text(query)
                    query_vector = self.embedder.embed_text(text)
            elif isinstance(query, Image.Image):
                query_vector = self.embedder.embed_image(query)
            else:
                raise ValueError("Unsupported query type")
        # Now perform similarity search in the vector index
        results = self.index.search(query_vector, top_k=top_k)
        # Format results: perhaps return documents with scores
        formatted_results = []
        for doc_ref, score in results:
            # doc_ref contains {"_id": ..., "_source": ..., "_field": ...}
            result_entry = {
                "_id": doc_ref["_id"],
                "_score": score,
                "_source": doc_ref["_source"],
                "_matched_field": doc_ref["_field"],
            }
            formatted_results.append(result_entry)
        return formatted_results
