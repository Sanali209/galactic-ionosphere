from loguru import logger
from src.core.database.models.task import TaskRecord
from src.core.database.models.image import ImageRecord
from src.core.database.models.detection import Detection
from src.core.ai.service import EmbeddingService
from src.core.ai.vector_driver import VectorDriver
from src.core.ai.detection import ObjectDetectionService
import uuid

class AITaskHandlers:
    """
    Handlers for AI-related tasks.
    """
    def __init__(self, vector_driver: VectorDriver, embedding_service: EmbeddingService, detection_service: ObjectDetectionService):
        self.vector_driver = vector_driver
        self.embedding_service = embedding_service
        self.detection_service = detection_service

    async def handle_generate_vectors(self, task: TaskRecord) -> dict:
        """
        Handler for 'GENERATE_VECTORS' task.
        Payload: {"image_id": str}
        """
        image_id = task.payload.get("image_id")
        if not image_id:
            raise ValueError("Missing image_id")

        # 1. Fetch Image
        image = await ImageRecord.get(image_id)
        if not image:
            raise ValueError(f"Image not found: {image_id}")
            
        full_path = image.full_path # Using property
        
        # 2. Generate Embedding
        # Ensure model loaded
        await self.embedding_service.load()
        vector = await self.embedding_service.encode_image(full_path)
        
        if not vector:
            raise RuntimeError("Failed to generate embedding (model error or empty)")

        # 3. Upsert to Qdrant
        oid_uuid = self.vector_driver.to_qdrant_id(str(image.id))
        
        # Payload for filterable search
        payload = {
            "mongo_id": str(image.id),
            "rating": image.rating,
            "tags": [str(t) for t in image.tag_ids]
        }
        
        await self.vector_driver.upsert_vector(
            point_id=oid_uuid,
            vector=vector,
            payload=payload
        )
        
        logger.info(f"Vector generated for {image.id}")
        return {"vector_len": len(vector)}

    async def handle_detect_objects(self, task: TaskRecord) -> dict:
        """
        Handler for 'DETECT_OBJECTS' task.
        Payload: {"image_id": str}
        """
        image_id = task.payload.get("image_id")
        if not image_id:
            raise ValueError("Missing image_id")

        image = await ImageRecord.get(image_id)
        if not image:
            raise ValueError(f"Image not found: {image_id}")

        # Run Detection
        results = await self.detection_service.detect(image.full_path)
        
        # Save Detections
        created_count = 0
        for res in results:
            det = Detection(
                parent_image_id=image.id,
                box=res["box"],
                class_label=res["label"],
                confidence=res["conf"]
            )
            await det.save()
            created_count += 1
            
        logger.info(f"Detected {created_count} objects in {image.id}")
        return {"detections": created_count}
