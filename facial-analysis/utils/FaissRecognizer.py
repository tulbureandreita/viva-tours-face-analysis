import faiss
import os
import numpy as np
import cv2
import uuid
import json

class FaissRecognizer:
    def __init__(self, dir_path: str, threshold: float, sdim: int = 512):
        self.dir_path = dir_path
        self.threshold = threshold
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Get the parent directory (.../facial-analysis)
        parent_dir = os.path.dirname(script_dir)

        # Create absolute paths in the parent directory
        self.index_path = os.path.join(parent_dir, "faiss.index")
        self.map_path = os.path.join(parent_dir, "id_map.json")

        # Initialize attributes before loading
        self.index = None
        self.next_id = 0
        self.id_to_uuid = {}
        self.uuid_to_id = {}

        # attempt loading
        self._load()

        # If loading fails or no index exists, initialize a new one
        if self.index is None:
            print("Initializing a new FAISS index.")
            flat = faiss.IndexFlatIP(sdim)
            self.index = faiss.IndexIDMap(flat)

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
            person_uuid = self.id_to_uuid[str(best_match_internal_id)]
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
        self.id_to_uuid[str(internal_id)] = person_uuid
        self.uuid_to_id[person_uuid] = internal_id
        self.next_id += 1

        os.makedirs(os.path.join(self.dir_path, f"{person_uuid}"), exist_ok=True)
        print(f"No good match found. Creating new person with ID: {person_uuid}")

        # Save the updated index and map every time a new person is added
        self._save()

        # When a new person is created, the similarity is effectively 1.0 to itself,
        # but returning 0.0 indicates it's a new entry.
        return person_uuid, 0.0

    def save_image(self, person_id: str, original_img: np.ndarray, basename: str):
        out_path = os.path.join(self.dir_path, f"{person_id}", f"{basename}.jpg")
        cv2.imwrite(out_path, original_img)

    def _load(self):
        """Loads the index and map from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.map_path):
            try:
                print(f"Loading FAISS index from {self.index_path}")
                self.index = faiss.read_index(self.index_path)

                print(f"Loading ID map from {self.map_path}")
                with open(self.map_path, 'r') as f:
                    self.id_to_uuid = json.load(f)

                # Rebuild the reverse map and find the next available ID
                self.uuid_to_id = {v: int(k) for k, v in self.id_to_uuid.items()}
                if self.id_to_uuid:
                    self.next_id = max(int(k) for k in self.id_to_uuid.keys()) + 1
                else:
                    self.next_id = 0
                print("Successfully loaded persistent data.")
            except Exception as e:
                print(f"Failed to load persistent data: {e}. Starting fresh.")
                # Clear potentially partially loaded data
                delattr(self, 'index') if hasattr(self, 'index') else None
        else:
            print("No persistent data found. Starting fresh.")


    def _save(self):
        """Saves the index and map to disk."""
        try:
            print(f"Saving FAISS index to {self.index_path}")
            faiss.write_index(self.index, self.index_path)

            print(f"Saving ID map to {self.map_path}")
            with open(self.map_path, 'w') as f:
                json.dump(self.id_to_uuid, f, indent=4)
        except Exception as e:
            print(f"Failed to save persistent data: {e}")