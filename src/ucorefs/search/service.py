"""
UCoreFS - Search Service

Unified search combining MongoDB text/filter queries with FAISS vector similarity.
Single entry point for all file search operations.
"""
from typing import List, Dict, Any, Optional, Tuple, Set
import re
from dataclasses import dataclass, field
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem


@dataclass(slots=True)
class SearchResult:
    """Individual search result with scoring."""
    file_id: ObjectId
    score: float = 1.0
    vector_score: Optional[float] = None
    text_score: Optional[float] = None
    match_type: str = "filter"  # filter, text, vector, hybrid


@dataclass(slots=True)
class SearchQuery:
    """
    Unified search query parameters.
    
    Supports:
    - Text search (file name, description, tags)
    - MongoDB filters (file_type, rating, label, etc.)
    - Vector similarity (CLIP or BLIP embeddings)
    """
    # Text query
    text: Optional[str] = None
    
    # MongoDB filters
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Vector search
    vector_search: bool = False
    vector_provider: str = "clip"  # clip, blip, mobilenet
    vector_query: Optional[List[float]] = None  # Pre-computed embedding
    
    # Pagination
    limit: int = 100
    offset: int = 0
    
    # Sorting
    sort_by: str = "score"  # score, name, modified_at, rating
    sort_desc: bool = True
    
    # Detection filters (parsed from query)
    # List of {"class_name": str, "group_name": str, "min_count": int, "negate": bool}
    detection_filters: List[Dict[str, Any]] = field(default_factory=list)


class SearchService(BaseSystem):
    """
            vector_provider="clip"
        )
        results = await search_service.search(query)
    """
    
    async def initialize(self) -> None:
        """Initialize search service."""
        logger.info("SearchService initializing")
        
        # Get dependencies
        self._faiss_service = None
        self._vector_service = None
        
        try:
            from src.ucorefs.vectors.faiss_service import FAISSIndexService
            self._faiss_service = self.locator.get_system(FAISSIndexService)
        except (KeyError, ImportError):
            logger.info("FAISSIndexService not available - vector search disabled")
        
        try:
            from src.ucorefs.vectors.service import VectorService
            self._vector_service = self.locator.get_system(VectorService)
        except (KeyError, ImportError):
            pass
        
        await super().initialize()
        logger.info("SearchService ready")
    
    async def shutdown(self) -> None:
        """Shutdown search service."""
        logger.info("SearchService shutting down")
        await super().shutdown()
    
    async def search(self, query: SearchQuery) -> List[SearchResult]:
        """
        Execute unified search.
        
        Args:
            query: SearchQuery with all parameters
            
        Returns:
            List of SearchResult sorted by score
        """
        from src.ucorefs.models.file_record import FileRecord
        
        logger.info(f"[SearchService] ========== SEARCH START ==========")
        logger.info(f"[SearchService] Text: '{query.text}'")
        logger.info(f"[SearchService] Vector Search: {query.vector_search}")
        logger.info(f"[SearchService] Vector Provider: {query.vector_provider}")
        logger.info(f"[SearchService] Filters: {query.filters}")
        logger.info(f"[SearchService] Limit: {query.limit}")
        
        results_map: Dict[str, SearchResult] = {}
        
        # Step 0: Parse detection syntax from text
        if query.text:
            query.text, parsed_filters = self._parse_query_text(query.text)
            if parsed_filters:
                query.detection_filters.extend(parsed_filters)
                logger.info(f"[SearchService] Parsed detection filters: {parsed_filters}")
        
        # Step 1: MongoDB filter search
        mongo_filter = self._build_mongo_filter(query)
        
        # Apply detection filters
        if query.detection_filters:
            detection_ids, excluded_ids = await self._resolve_detection_filters(query.detection_filters)
            
            # Combine logic
            if detection_ids:
                mongo_filter["_id"] = {"$in": list(detection_ids)}
            
            if excluded_ids:
                # If we already have $in, we subtract excluded
                if "_id" in mongo_filter and "$in" in mongo_filter["_id"]:
                    current_in = set(mongo_filter["_id"]["$in"])
                    current_in -= excluded_ids
                    mongo_filter["_id"]["$in"] = list(current_in)
                else:
                    # Just exclude
                    if "_id" not in mongo_filter:
                         mongo_filter["_id"] = {}
                    mongo_filter["_id"]["$nin"] = list(excluded_ids)
        
        if query.text:
            # Text search on name, description, tags
            mongo_filter["$or"] = [
                {"name": {"$regex": query.text, "$options": "i"}},
                {"description": {"$regex": query.text, "$options": "i"}},
                {"ai_description": {"$regex": query.text, "$options": "i"}},
            ]
        
        
        # Clean up filter - remove empty lists that custom ORM might not handle well
        mongo_filter_clean = {}
        for key, value in mongo_filter.items():
            # Skip empty list values
            if isinstance(value, list) and len(value) == 0:
                continue
            # Keep everything else
            mongo_filter_clean[key] = value
        
        logger.info(f"[SearchService] MongoDB Filter (cleaned): {mongo_filter_clean}")
        
        # Execute MongoDB query
        mongo_limit = query.limit * 2 if query.vector_search else query.limit
        logger.info(f"[SearchService] Executing MongoDB query (limit={mongo_limit})...")
        
        files = await FileRecord.find(
            mongo_filter_clean,
            limit=mongo_limit
        )
        
        logger.info(f"[SearchService] MongoDB returned {len(files)} FileRecords")
        
        # Debug: Check if detection filter IDs exist
        if query.detection_filters and "_id" in mongo_filter and "$in" in mongo_filter["_id"]:
            requested_ids = mongo_filter["_id"]["$in"]
            logger.info(f"[SearchService] Requested {len(requested_ids)} IDs from detection filter")
            if len(files) == 0 and len(requested_ids) > 0:
                logger.warning(f"[SearchService] Detection filter found {len(requested_ids)} IDs but FileRecord.find returned 0!")
                logger.warning(f"[SearchService] Sample requested IDs: {requested_ids[:3]}")
                # Check if these IDs actually exist as FileRecords
                test_file = await FileRecord.find({"_id": requested_ids[0]}, limit=1)
                logger.warning(f"[SearchService] Direct query for first ID returned: {len(test_file)} files")
        
        # Add to results
        for file in files:
            file_id_str = str(file._id)
            text_score = 1.0
            
            # Boost exact name matches
            if query.text and query.text.lower() in file.name.lower():
                text_score = 1.5
            
            results_map[file_id_str] = SearchResult(
                file_id=file._id,
                score=text_score,
                text_score=text_score,
                match_type="text" if query.text else "filter"
            )
        
        logger.info(f"[SearchService] Initial results_map has {len(results_map)} entries")
        
        # Step 2: Vector similarity search
        if query.vector_search:
            if not self._faiss_service:
                logger.warning(f"[SearchService] Vector search requested but FAISS service NOT AVAILABLE")
                logger.warning(f"[SearchService] This is likely why you're getting same results as text search!")
            else:
                logger.info(f"[SearchService] ✓ FAISS service available, executing vector search")
                vector_results = await self._vector_search(query, list(results_map.keys()))
                
                logger.info(f"[SearchService] Vector search returned {len(vector_results)} results")
                
                # Merge vector scores
                for file_id, vector_score in vector_results:
                    file_id_str = str(file_id)
                    
                    if file_id_str in results_map:
                        # Hybrid: combine text and vector scores
                        existing = results_map[file_id_str]
                        existing.vector_score = vector_score
                        existing.score = (existing.text_score or 1.0) * 0.4 + vector_score * 0.6
                        existing.match_type = "hybrid"
                        logger.debug(f"[SearchService] Hybrid result: {file_id_str[:8]}... "
                                   f"text_score={existing.text_score:.4f}, vector_score={vector_score:.4f}, "
                                   f"final_score={existing.score:.4f}")
                    else:
                        # Vector-only result
                        results_map[file_id_str] = SearchResult(
                            file_id=file_id,
                            score=vector_score,
                            vector_score=vector_score,
                            match_type="vector"
                        )
                        logger.debug(f"[SearchService] Vector-only result: {file_id_str[:8]}... score={vector_score:.4f}")
                
                logger.info(f"[SearchService] After vector merge: {len(results_map)} total results")
        else:
            logger.info(f"[SearchService] Skipping vector search (vector_search={query.vector_search})")
        
        # Step 3: Sort and paginate
        results = list(results_map.values())
        
        logger.info(f"[SearchService] Sorting {len(results)} results by {query.sort_by}")
        
        if query.sort_by == "score":
            results.sort(key=lambda r: r.score, reverse=query.sort_desc)
        
        # Log top results
        if results:
            logger.info(f"[SearchService] Top 5 results:")
            for i, r in enumerate(results[:5]):
                logger.info(f"[SearchService]   {i+1}. file_id={str(r.file_id)[:12]}... score={r.score:.4f} "
                           f"type={r.match_type} vector={r.vector_score} text={r.text_score}")
        
        # Apply offset/limit
        final_results = results[query.offset:query.offset + query.limit]
        logger.info(f"[SearchService] ========== SEARCH COMPLETE: {len(final_results)} results ==========")
        return final_results
    
    async def search_similar(
        self,
        file_id: ObjectId,
        provider: str = "clip",
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Find files similar to a given file.
        
        Args:
            file_id: Source file ObjectId
            provider: Embedding provider
            limit: Max results
            
        Returns:
            List of similar files
        """
        if not self._faiss_service:
            return []
        
        try:
            from src.ucorefs.vectors.models import EmbeddingRecord
            
            # Get source embedding
            embedding = await EmbeddingRecord.find_one({
                "file_id": file_id,
                "provider": provider
            })
            
            if not embedding:
                logger.debug(f"No {provider} embedding for {file_id}")
                return []
            
            # Search similar
            similar = await self._faiss_service.search(
                provider,
                embedding.vector,
                k=limit + 1  # +1 to exclude self
            )
            
            # Filter out source file and format results
            results = []
            for fid, score in similar:
                if fid != file_id:
                    results.append(SearchResult(
                        file_id=fid,
                        score=score,
                        vector_score=score,
                        match_type="vector"
                    ))
            
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Similar search failed: {e}")
            return []
    
    async def _vector_search(
        self,
        query: SearchQuery,
        filtered_ids: List[str]
    ) -> List[tuple]:
        """Execute vector similarity search."""
        logger.info(f"[SearchService._vector_search] Starting vector search")
        
        if not self._faiss_service:
            logger.error(f"[SearchService._vector_search] FAISS service is None - cannot execute vector search!")
            return []
        
        logger.info(f"[SearchService._vector_search] FAISS service available")
        
        # Get query embedding
        query_vector = query.vector_query
        
        if not query_vector and query.text:
            # Generate embedding from text (requires embedding service)
            logger.info(f"[SearchService._vector_search] No pre-computed embedding, generating from text: '{query.text}'")
            query_vector = await self._get_text_embedding(query.text, query.vector_provider)
            
            if query_vector:
                logger.info(f"[SearchService._vector_search] ✓ Generated embedding vector (dim={len(query_vector)})")
            else:
                logger.error(f"[SearchService._vector_search] ✗ Failed to generate text embedding")
        
        if not query_vector:
            logger.error(f"[SearchService._vector_search] No query vector available - cannot search!")
            return []
        
        logger.info(f"[SearchService._vector_search] Query vector ready (dim={len(query_vector)})")
        
        # Search with optional filtering
        file_ids = [ObjectId(fid) for fid in filtered_ids] if filtered_ids else None
        logger.info(f"[SearchService._vector_search] Searching with filter: {len(file_ids) if file_ids else 0} file IDs")
        
        logger.info(f"[SearchService._vector_search] Calling FAISS.search(provider='{query.vector_provider}', k={query.limit})")
        
        try:
            results = await self._faiss_service.search(
                query.vector_provider,
                query_vector,
                k=query.limit,
                file_ids=file_ids
            )
            logger.info(f"[SearchService._vector_search] ✓ FAISS returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"[SearchService._vector_search] ✗ FAISS search failed: {e}")
            import traceback
            logger.error(f"[SearchService._vector_search] Traceback: {traceback.format_exc()}")
            return []
    
    async def _get_text_embedding(self, text: str, provider: str) -> Optional[List[float]]:
        """Get embedding for text query using CLIPExtractor."""
        logger.info(f"[SearchService._get_text_embedding] Generating embedding for text: '{text}' using {provider}")
        
        if provider == "clip":
            try:
                from src.ucorefs.extractors.clip_extractor import CLIPExtractor
                
                logger.info(f"[SearchService._get_text_embedding] Initializing CLIPExtractor...")
                extractor = CLIPExtractor(self.locator)
                
                logger.info(f"[SearchService._get_text_embedding] Encoding text with CLIP...")
                embedding = await extractor.encode_text(text)
                
                if embedding:
                    logger.info(f"[SearchService._get_text_embedding] ✓ Successfully encoded text (dim={len(embedding)})")
                    return embedding
                else:
                    logger.warning(f"[SearchService._get_text_embedding] ✗ CLIP encoding returned None")
            except Exception as e:
                logger.error(f"[SearchService._get_text_embedding] ✗ CLIP text encoding failed: {e}")
                import traceback
                logger.error(f"[SearchService._get_text_embedding] Traceback: {traceback.format_exc()}")
        else:
            logger.warning(f"[SearchService._get_text_embedding] Unsupported provider: {provider}")
        
        logger.debug(f"[SearchService._get_text_embedding] Text embedding not available for {provider}")
        return None
    
    def _build_mongo_filter(self, query: SearchQuery) -> Dict[str, Any]:
        """Build MongoDB filter from query parameters."""
        mongo_filter = {}
        
        for key, value in query.filters.items():
            if value is not None:
                if key == "tag_ids" and isinstance(value, list):
                    mongo_filter["tag_ids"] = {"$in": value}
                elif key == "rating_min":
                    mongo_filter["rating"] = {"$gte": value}
                elif key == "file_types" and isinstance(value, list):
                    mongo_filter["file_type"] = {"$in": value}
                else:
                    mongo_filter[key] = value
        
        return mongo_filter
    
    async def get_detection_counts(self, query: SearchQuery = None) -> List[Dict[str, Any]]:
        """
        Get aggregated detection counts for display in the tree.
        
        Args:
            query: Current active search query (to filter counts). 
                   If None, returns global counts.
                   
        Returns:
            List of dicts: [{'class_name': 'Person', 'group_name': 'any', 'count': 10}, ...]
        """
        try:
            from src.ucorefs.detection.models import DetectionInstance, DetectionClass
            
            # Base match pipeline
            pipeline = []
            
            # Scope to current search query
            if query:
                # 1. Build file scoping filter
                file_filter = self._build_mongo_filter(query)
                
                # Handling Text Search separately (since it's not in mongo_filter usually? 
                # Wait, _build_mongo_filter handles 'filters', but search handles 'text' logic separately in `search` method)
                
                # We need to replicate logic from search() to gather candidate file IDs
                # This is slightly redundant but necessary for accurate counts
                
                # Check if we need scoping (if filter non-empty or text present)
                has_filter = bool(file_filter)
                has_text = bool(query.text)
                
                if has_filter or has_text:
                    # Resolve text search filter parts if needed
                    # Note: search() does this inside it. We might duplicating logic.
                    # Simplified approach: Just use file_filter as base, ignore text relevance scoring, just regex match.
                    
                    if has_text:
                         file_filter["$or"] = [
                            {"name": {"$regex": query.text, "$options": "i"}},
                            {"description": {"$regex": query.text, "$options": "i"}},
                            {"ai_description": {"$regex": query.text, "$options": "i"}},
                        ]
                    
                    # Fetch matching file IDs
                    # Cap search to avoid massive fetch
                    from src.ucorefs.models.file_record import FileRecord
                    
                    # Use custom ORM find() API with limit parameter
                    matching_files = await FileRecord.find(file_filter, limit=10000)
                    matching_ids = [f.id for f in matching_files]
                    
                    if not matching_ids:
                        return []
                        
                    # Add to pipeline
                    pipeline.append({
                        "$match": {"file_id": {"$in": matching_ids}}
                    })
            
            # Group by class_id AND group_name for subgroups
            pipeline.append({
                "$group": {
                    "_id": {
                        "class_id": "$detection_class_id",
                        "group": "$group_name"
                    },
                    "count": {"$sum": 1}
                }
            })
            
            # Execute aggregation
            cursor = DetectionInstance.get_collection().aggregate(pipeline)
            
            counts = []
            # Map (class_id, group) -> count
            group_counts = {}
            class_ids_set = set()
            
            async for doc in cursor:
                id_obj = doc["_id"]
                c_id = id_obj.get("class_id")
                g_name = id_obj.get("group", "unknown")
                
                if c_id:
                    class_ids_set.add(c_id)
                    group_counts[(c_id, g_name)] = doc["count"]
            
            if not class_ids_set:
                return []
            
            # Resolve class names
            classes = await DetectionClass.find({"_id": {"$in": list(class_ids_set)}})
            class_map = {cls.id: cls.class_name for cls in classes}
            
            # Build result with groups
            for (class_id, group_name), count in group_counts.items():
                class_name = class_map.get(class_id, "Unknown")
                counts.append({
                    "class_name": class_name,
                    "group_name": group_name,
                    "count": count,
                    "class_id": str(class_id)
                })
            
            return counts
            
        except Exception as e:
            logger.error(f"[SearchService] Failed to get detection counts: {e}")
            return []
    
    def _parse_query_text(self, text: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Parse detection syntax from text query.
        Syntax: [-!]class[:group][:count]
        Examples: 
          "Person:2" -> Class=Person, Count=2
          "Person:face" -> Class=Person, Group=face
          "Person:face:2" -> Class=Person, Group=face, Count=2
          "!Car" -> Class=Car, Negate=True
        """
        if not text:
            return "", []
        
        filters = []
        
        # Regex to find potential tokens
        # Matches: Optional -/!, Word, Optional :suffix, Optional :suffix
        # We capture the entire token and parse manually
        pattern = r'(?:^|\s)([-!]?[\w]+(?:[:][\w]+){0,2})(?=\s|$)'
        
        matches = list(re.finditer(pattern, text))
        
        last_end = 0
        clean_parts = []
        
        for match in matches:
            token = match.group(1)
            
            # Skip if it's just a regular word (no negation, no colons)
            if ':' not in token and not token.startswith(('!', '-')):
                # It's just a word like "vacation" or "Person"
                # Keep it as text
                continue
            
            # Parse token
            negate = False
            if token.startswith(('!', '-')):
                negate = True
                token = token[1:]
            
            parts = token.split(':')
            class_name = parts[0]
            group_name = "any"
            min_count = 1
            
            if len(parts) == 2:
                # Class:Suffix
                # Check if suffix is integer
                if parts[1].isdigit():
                    min_count = int(parts[1])
                else:
                    group_name = parts[1]
            elif len(parts) == 3:
                # Class:Group:Count
                # Middle is group, last is count (if digit)
                if parts[2].isdigit():
                    min_count = int(parts[2])
                    group_name = parts[1]
                else:
                    # Class:Group:SubGroup? Not supported yet, treat as Class:Group
                    group_name = parts[1]
            
            filters.append({
                "class_name": class_name,
                "group_name": group_name,
                "min_count": min_count,
                "negate": negate
            })
            
            # Add text before this match
            clean_parts.append(text[last_end:match.start()])
            last_end = match.end()
        
        # Add remaining text
        clean_parts.append(text[last_end:])
        
        cleaned_text = "".join(clean_parts).strip()
        # Collapse multiple spaces
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        
        return cleaned_text, filters

    async def _resolve_detection_filters(self, filters: List[Dict[str, Any]]) -> Tuple[Set[ObjectId], Set[ObjectId]]:
        """
        Resolve detection filters to Sets of File IDs.
        Returns: (included_ids, excluded_ids)
        """
        included_ids: Set[ObjectId] = set()
        excluded_ids: Set[ObjectId] = set()
        
        try:
            from src.ucorefs.detection.models import DetectionInstance
            
            # Optimization: If no filters, return empty
            if not filters:
                return set(), set()
                
            first_include = True
            
            for f in filters:
                class_name = f['class_name']
                group_name = f['group_name']
                min_count = f['min_count']
                negate = f['negate']
                
                # Build MongoDB query for DetectionInstance
                # We need to find file_ids where count(class matching) >= min_count
                
                # Resolving Class Name to ID? 
                # Currently detections might store class_name string if loose, or we assume lookup.
                # For Phase 4.1 merger we used 'label'.
                # DetectionInstance has 'detection_class_id'.
                # We need to lookup class ID from name first?
                # Or does DetectionInstance also store 'class_name' cached? 
                # Reading models.py: DetectionInstance has class_id and object_id. NOT class_name string (except my mock plan).
                # WAIT! DetectionInstance in Phase 4.1 Plan mentioned "class_name" in find logic? 
                # Let's check models.py again.
                # DetectionClass has class_name.
                
                # We need to find DetectionClass by name first.
                # This makes it async complex.
                
                # For this implementation step, let's assume we can query by 'class_name' field 
                # if we denormalized it, OR we do the lookup.
                # Given 'DetectionInstance' definition I checked earlier (Step 641), 
                # it only has 'detection_class_id'. It does NOT have 'class_name'.
                # So we MUST lookup class_id.
                
                # However, for simplicity and performance, maybe we filter by text match on FileRecord.detections?
                # NO, the plan is to use DetectionInstance.
                
                # Logic:
                # 1. Find Class ID(s) matching current class_name.
                # 2. Query DetectionInstance for these class_ids.
                
                # Mocking this logical flow for now as "todo" logic 
                # since we don't have the Class Lookup Service ready/injected here.
                # Wait, I can import DetectionClass model.
                
                from src.ucorefs.detection.models import DetectionClass
                
                # Find class(es)
                # Regex for case insensitive
                class_docs = await DetectionClass.find({"class_name": {"$regex": f"^{class_name}$", "$options": "i"}})
                if not class_docs:
                    logger.debug(f"Class '{class_name}' not found")
                    # If negating non-existent class -> exclude nothing.
                    # If including -> return empty set (intersection will empty result).
                    if not negate:
                        if first_include:
                            return set(), set() # Empty result immediately
                        else:
                            included_ids = set() # Intersect with empty -> empty
                    continue
                
                class_ids = [c.id for c in class_docs]
                
                # Query Instances
                # We need aggregation to count?
                # Or just simple query: find instances with class_in [...]
                # Then python count?
                
                # Aggregation is better:
                # pipeline = [
                #   {$match: {detection_class_id: {$in: class_ids}}},
                #   {$group: {_id: "$file_id", count: {$sum: 1}}},
                #   {$match: {count: {$gte: min_count}}}
                # ]
                
                pipeline = [
                    {"$match": {"detection_class_id": {"$in": class_ids}}},
                    {"$group": {"_id": "$file_id", "count": {"$sum": 1}}},
                    {"$match": {"count": {"$gte": min_count}}}
                ]
                
                
                # Execute aggregation using DatabaseManager
                from src.core.database.manager import DatabaseManager
                db_manager = self.locator.get_system(DatabaseManager)
                collection = db_manager.db[DetectionInstance._collection_name]
                cursor = collection.aggregate(pipeline)
                found_ids = set()
                async for doc in cursor:
                    found_ids.add(doc['_id'])
                
                logger.info(f"[DETECTION_FILTER] Aggregation found {len(found_ids)} file IDs for class_name='{class_name}': {list(found_ids)[:10]}")
                
                if negate:
                    excluded_ids.update(found_ids)
                else:
                    if first_include:
                        included_ids = found_ids
                        first_include = False
                    else:
                        included_ids.intersection_update(found_ids)
            
            logger.info(f"[DETECTION_FILTER] Final included_ids: {len(included_ids)} IDs")
            logger.info(f"[DETECTION_FILTER] Final excluded_ids: {len(excluded_ids)} IDs")
            if included_ids:
                logger.info(f"[DETECTION_FILTER] Sample included IDs: {list(included_ids)[:5]}")
            
            return included_ids, excluded_ids

        except Exception as e:
            logger.error(f"Failed to resolve detection filters: {e}")
            return set(), set()
