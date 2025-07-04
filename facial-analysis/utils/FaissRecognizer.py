import faiss
import os
import numpy as np
import cv2
import uuid
import json

class FaissRecognizer:
    def __init__(self, dir_path: str, threshold: float, sdim: int = 512):
        flat = faiss.IndexFlatIP(sdim)
        self.index = faiss.IndexIDMap(flat)
        self.next_id = 0
        self.dir_path = dir_path
        self.threshold = threshold
        self.index_path = "faiss.index"
        self.map_path = "id_map.json"
        self.id_to_uuid = {}
        self.uuid_to_id = {}

    def recognize_and_assign(self, embeddings: list[np.ndarray]):
        """
        Takes a list of face embeddings from a single image, finds the best match
        in the database, and assigns the image to that person. If no good match
        is found, creates a new person.
        """
        if not embeddings:
            return None, 0.0

        # If the database is empty, create a new person with the first embedding
        if self.index.ntotal == 0:
            return self._create_new_person(embeddings[0])

        best_overall_similarity = -1.0
        best_match_internal_id = -1

        # Search for the best match for each embedding in the image
        for emb in embeddings:
            x = emb.astype('float32').reshape(1, -1)
            # D is similarity, I is the internal_id
            D, I = self.index.search(x, 1)
            similarity = D[0, 0]
            internal_id = I[0, 0]

            if similarity > best_overall_similarity:
                best_overall_similarity = similarity
                best_match_internal_id = internal_id

        print(
            f"Best match candidate: ID {best_match_internal_id} with similarity {best_overall_similarity:.4f} (Threshold: {self.threshold})")

        # Decision: Is the best match good enough?
        if best_overall_similarity > self.threshold:
            # Yes, it's a match. Return the existing person's UUID.
            person_uuid = self.id_to_uuid[best_match_internal_id]
            return person_uuid, best_overall_similarity
        else:
            # No, no good match found. Create a new person using the first embedding.
            return self._create_new_person(embeddings[0])

    def _create_new_person(self, embedding: np.ndarray):
        """Creates a new person, adds them to the index, and returns their ID."""
        internal_id = self.next_id
        person_uuid = str(uuid.uuid4())

        x = embedding.astype('float32').reshape(1, -1)
        self.index.add_with_ids(x, np.array([internal_id], dtype='int64'))
        self.id_to_uuid[internal_id] = person_uuid
        self.uuid_to_id[person_uuid] = internal_id
        self.next_id += 1

        os.makedirs(os.path.join(self.dir_path, f"{person_uuid}"), exist_ok=True)
        print(f"No good match found. Creating new person with ID: {person_uuid}")

        # When a new person is created, the similarity is effectively 1.0 to itself,
        # but returning 0.0 indicates it's a new entry.
        return person_uuid, 0.0

    def save_image(self, person_id: str, original_img: np.ndarray, basename: str):
        out_path = os.path.join(self.dir_path, f"{person_id}", f"{basename}.jpg")
        cv2.imwrite(out_path, original_img)
