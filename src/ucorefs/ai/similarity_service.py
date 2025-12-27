"""
UCoreFS - Similarity Service

Background service for finding and creating similarity relations.
"""
from typing import List
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.vectors.service import VectorService
from src.ucorefs.models.file_record import FileRecord


class SimilarityService(BaseSystem):
    """
    Service for finding similar files using vector embeddings.
    
    Automatically creates relation records between similar files.
    Runs in background via TaskSystem integration.
    """
    
    async def initialize(self) -> None:
        """Initialize similarity service."""
        logger.info("SimilarityService initializing")
        
        # Get dependencies
        self.vector_service = self.locator.get_system(VectorService)
        
        # Get configuration
        self.default_threshold = 0.85
        if hasattr(self.config.data, 'similarity'):
            self.default_threshold = self.config.data.similarity.threshold
        
        await super().initialize()
        logger.info(f"SimilarityService ready (threshold: {self.default_threshold})")
    
    async def shutdown(self) -> None:
        """Shutdown similarity service."""
        logger.info("SimilarityService shutting down")
        await super().shutdown()
    
    async def find_and_create_relations(
        self,
        file_id: ObjectId,
        threshold: float = None,
        limit: int = 10
    ) -> int:
        """
        Find similar files and create relation records.
        
        Args:
            file_id: Source file ObjectId
            threshold: Similarity threshold (0.0-1.0)
            limit: Max similar files to find
            
        Returns:
            Number of relations created
        """
        threshold = threshold or self.default_threshold
        
        try:
            # Get file record
            file = await FileRecord.get(file_id)
            if not file or not file.has_vector:
                logger.debug(f"File {file_id} has no vector, skipping")
                return 0
            
            # Get file's embedding from ChromaDB
            # Note: In real implementation, we'd get the actual vector
            # For now, this is a placeholder
            
            if not self.vector_service.is_available():
                logger.warning("VectorService unavailable")
                return 0
            
            # Search for similar files
            # This would use the file's actual vector
            # results = await self.vector_service.search(
            #     "file_embeddings",
            #     file_vector,
            #     filters={"file_type": file.file_type},
            #     limit=limit
            # )
            
            # For now, placeholder
            logger.info(f"Similarity search for {file_id} (threshold: {threshold})")
            
            # Create relations for each similar file
            relations_created = 0
            # for result in results:
            #     if result["score"] >= threshold:
            #         await self._create_relation(
            #             file_id,
            #             result["file_id"],
            #             "image-image",
            #             "near_duplicate",
            #             {"score": result["score"], "threshold": threshold}
            #         )
            #         relations_created += 1
            
            logger.info(f"Created {relations_created} relations for {file_id}")
            return relations_created
        
        except Exception as e:
            logger.error(f"Failed to find similar files: {e}")
            return 0
    
    async def find_similar(
        self,
        file_id: ObjectId,
        provider: str = "clip",
        threshold: float = None,
        limit: int = 10
    ) -> List[dict]:
        """
        Find files similar to the given file.
        
        Args:
            file_id: Source file ObjectId
            provider: Embedding provider (clip, blip, mobilenet)
            threshold: Minimum similarity score (0.0-1.0)
            limit: Max results
            
        Returns:
            List of dicts with file_id, score, and file record
        """
        from src.ucorefs.models.file_record import FileRecord
        from src.ucorefs.vectors.models import EmbeddingRecord
        
        threshold = threshold or self.default_threshold
        
        try:
            # Get source file
            file = await FileRecord.get(file_id)
            if not file:
                return []
            
            # Get embedding from EmbeddingRecord ORM (correct source)
            embedding = await EmbeddingRecord.find_one({
                "file_id": file_id,
                "provider": provider
            })
            
            if not embedding or not embedding.vector:
                logger.debug(f"File {file_id} has no {provider} embedding in EmbeddingRecord")
                return []
            
            vector = embedding.vector
            
            # Search via VectorService  
            if not self.vector_service.is_available():
                logger.warning("VectorService not available for similarity search")
                return []
            
            # Search similar vectors
            results = await self.vector_service.search(
                collection=provider,
                query_vector=vector,
                filters={"file_type": file.file_type},  # Same type only
                limit=limit + 1  # +1 to exclude self
            )
            
            # Filter out self and below threshold
            similar_files = []
            for result in results:
                result_id = result.get("file_id")
                score = result.get("score", 0)
                
                # Skip self
                if str(result_id) == str(file_id):
                    continue
                
                # Apply threshold
                if score < threshold:
                    continue
                
                # Load file record
                similar_file = await FileRecord.get(ObjectId(result_id))
                if similar_file:
                    similar_files.append({
                        "file_id": result_id,
                        "score": score,
                        "file": similar_file
                    })
                
                if len(similar_files) >= limit:
                    break
            
            logger.debug(f"Found {len(similar_files)} similar files for {file_id}")
            return similar_files
            
        except Exception as e:
            logger.error(f"Failed to find similar files: {e}")
            return []
    
    async def _create_relation(
        self,
        source_id: ObjectId,
        target_id: ObjectId,
        relation_type: str,
        subtype: str,
        payload: dict
    ) -> bool:
        """
        Create a relation record.
        
        Args:
            source_id: Source file ID
            target_id: Target file ID
            relation_type: Type of relation (e.g., "file-file")
            subtype: Relation subtype (e.g., "near_duplicate", "exact_duplicate")
            payload: Additional data
            
        Returns:
            True if created successfully
        """
        try:
            from src.ucorefs.relations.service import RelationService
            
            relation_service = self.locator.get_system(RelationService)
            
            await relation_service.create_relation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                sub_type=subtype,
                payload=payload
            )
            
            logger.debug(
                f"Created relation: {source_id} -> {target_id} "
                f"({relation_type}/{subtype}, score: {payload.get('score')})"
            )
            return True
            
        except Exception as e:
            logger.error(f"Failed to create relation: {e}")
            return False
    
    # ==================== Duplicate Marking ====================
    
    async def mark_as_duplicate(
        self,
        file_id_1: ObjectId,
        file_id_2: ObjectId,
        duplicate_type: str = "near_duplicate"
    ) -> dict:
        """
        Mark two files as duplicates.
        
        Creates a bidirectional relation between the files.
        
        Args:
            file_id_1: First file ObjectId
            file_id_2: Second file ObjectId
            duplicate_type: Type of duplicate:
                - "exact_duplicate": Hash-verified identical
                - "near_duplicate": Visually similar (default)
                - "similar": Related but distinct
                - "same_set": Part of same sequence/set
                
        Returns:
            Dict with: success, relation_id, error (if any)
        """
        from datetime import datetime
        
        result = {
            "success": False,
            "file_id_1": str(file_id_1),
            "file_id_2": str(file_id_2),
            "duplicate_type": duplicate_type
        }
        
        try:
            from src.ucorefs.relations.service import RelationService
            
            relation_service = self.locator.get_system(RelationService)
            
            # Create relation
            relation = await relation_service.create_relation(
                source_id=file_id_1,
                target_id=file_id_2,
                relation_type="file-file",
                sub_type=duplicate_type,
                bidirectional=True,  # Both directions
                payload={
                    "marked_at": datetime.utcnow().isoformat(),
                    "marked_by": "user"
                }
            )
            
            if relation:
                result["success"] = True
                result["relation_id"] = str(relation.id) if hasattr(relation, "id") else None
                logger.info(f"Marked as {duplicate_type}: {file_id_1} <-> {file_id_2}")
            else:
                result["error"] = "Failed to create relation"
                
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Failed to mark as duplicate: {e}")
        
        return result
    
    async def get_duplicates(
        self,
        file_id: ObjectId,
        duplicate_type: str = None
    ) -> List[dict]:
        """
        Get all files marked as duplicates of the given file.
        
        Args:
            file_id: Source file ObjectId
            duplicate_type: Optional filter by type (exact_duplicate, near_duplicate, etc.)
            
        Returns:
            List of dicts with: file_id, duplicate_type, marked_at, file
        """
        try:
            from src.ucorefs.relations.service import RelationService
            
            relation_service = self.locator.get_system(RelationService)
            
            # Get relations where this file is source or target
            relations = await relation_service.get_relations(
                source_id=file_id,
                relation_type="file-file"
            )
            
            # Also get reverse relations
            reverse_relations = await relation_service.get_relations(
                target_id=file_id,
                relation_type="file-file"
            )
            
            # Combine and deduplicate
            all_relations = relations + reverse_relations
            
            duplicates = []
            seen_ids = set()
            
            for rel in all_relations:
                # Get the other file's ID
                other_id = rel.target_id if rel.source_id == file_id else rel.source_id
                
                if str(other_id) in seen_ids:
                    continue
                seen_ids.add(str(other_id))
                
                # Filter by type if specified
                if duplicate_type and rel.sub_type != duplicate_type:
                    continue
                
                # Load the other file
                other_file = await FileRecord.get(other_id)
                if other_file:
                    duplicates.append({
                        "file_id": str(other_id),
                        "duplicate_type": rel.sub_type,
                        "marked_at": rel.payload.get("marked_at") if rel.payload else None,
                        "file": other_file
                    })
            
            return duplicates
            
        except KeyError:
            logger.warning("RelationService not available")
            return []
        except Exception as e:
            logger.error(f"Failed to get duplicates: {e}")
            return []
    
    async def is_duplicate(
        self,
        file_id_1: ObjectId,
        file_id_2: ObjectId
    ) -> bool:
        """
        Check if two files are marked as duplicates.
        
        Args:
            file_id_1: First file ObjectId
            file_id_2: Second file ObjectId
            
        Returns:
            True if a duplicate relation exists
        """
        try:
            from src.ucorefs.relations.service import RelationService
            
            relation_service = self.locator.get_system(RelationService)
            
            # Check if relation exists
            exists = await relation_service.relation_exists(
                source_id=file_id_1,
                target_id=file_id_2,
                relation_type="file-file"
            )
            
            if exists:
                return True
            
            # Check reverse direction
            return await relation_service.relation_exists(
                source_id=file_id_2,
                target_id=file_id_1,
                relation_type="file-file"
            )
            
        except Exception as e:
            logger.error(f"Failed to check duplicate status: {e}")
            return False

