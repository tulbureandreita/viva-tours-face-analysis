import faiss
import os
import numpy as np
import cv2
import uuid
import json


class FaissRecognizer:
    def __init__(self, dir_path: str, threshold: float, sdim: int = 512):
        # Base index for L2; wrap in IDMap so we control the IDs
        # IDMap wraps IndexFlatL2 so we can use our own integer IDs
        flat = faiss.IndexFlatL2(sdim)
        self.index = faiss.IndexIDMap(flat)
        self.next_id = 0
        self.dir_path = dir_path
        self.threshold = threshold
        self.index_path = "faiss.index"
        self.map_path = "id_map.json"
        self.id_to_uuid = {}
        self.uuid_to_id = {}
        # Load existing index and map if present
        #self._load()

    def assign(self, embedding: np.ndarray, face_gender) -> uuid.uuid4():
        x = embedding.astype('float32').reshape(1,-1)
        Distance = 0.0
        print(self.index.ntotal, "persons in index")
        if self.index.ntotal > 0:
            Distance, Index = self.index.search(x, 1)
            print(f"NN dist={Distance[0, 0]}, (thr={self.threshold})")
            #TODO: implement gender check
            if Distance[0, 0] < self.threshold: #and face_gender = person.gender:
                internal_id = int(Index[0, 0])
                return self.id_to_uuid[internal_id], Distance
        internal_id = self.next_id
        person_uuid = str(uuid.uuid4())

        # new person: add to index
        self.index.add_with_ids(x, np.array([internal_id], dtype='int64'))
        self.id_to_uuid[internal_id] = person_uuid
        self.uuid_to_id[person_uuid] = internal_id
        self.next_id += 1
        # make folder for them
        os.makedirs(os.path.join(self.dir_path, f"{person_uuid}"), exist_ok=True)
        #self._save() # Save index and map after adding a new person if you need persistence

        return person_uuid, Distance

    def save_image(self, person_id: str, original_img: np.ndarray, basename: str):
        out_path = os.path.join(self.dir_path, f"{person_id}", f"{basename}.jpg")
        cv2.imwrite(out_path, original_img)

    def _load(self):
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if os.path.exists(self.map_path):
            with open(self.map_path, 'r') as f:
                self.id_to_uuid = json.load(f)
            # rebuild reverse map and next_id
            for internal_id, person_uuid in self.id_to_uuid.items():
                iid = int(internal_id)
                self.uuid_to_id[person_uuid] = iid
                self.next_id = max(self.next_id, iid+1)

    def _save(self):
        # Save FAISS index
        faiss.write_index(self.index, self.index_path)
        # Save mapping dict
        with open(self.map_path, 'w') as f:
            json.dump({str(k): v for k, v in self.id_to_uuid.items()}, f)