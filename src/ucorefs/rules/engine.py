"""
UCoreFS - Rules Engine

Engine for executing automation rules.
"""
from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from loguru import logger

from src.core.base_system import BaseSystem
from src.ucorefs.rules.models import Rule
from src.ucorefs.rules.conditions import create_condition
from src.ucorefs.rules.actions import create_action
from src.ucorefs.models.file_record import FileRecord


class RulesEngine(BaseSystem):
    """
    Rules execution engine.
    
    Features:
    - Evaluate rules on triggers
    - Execute actions when conditions match
    - Track execution statistics
    """
    
    async def initialize(self) -> None:
        """Initialize rules engine."""
        logger.info("RulesEngine initializing")
        await super().initialize()
        logger.info("RulesEngine ready")
    
    async def shutdown(self) -> None:
        """Shutdown rules engine."""
        logger.info("RulesEngine shutting down")
        await super().shutdown()
    
    async def execute_on_trigger(
        self,
        trigger: str,
        file: FileRecord,
        context: dict = None
    ) -> int:
        """
        Execute all enabled rules for a trigger.
        
        Args:
            trigger: Trigger type (on_import, on_tag, manual)
            file: FileRecord to evaluate
            context: Additional context
            
        Returns:
            Number of rules executed
        """
        # Get all enabled rules for this trigger
        rules = await Rule.find({
            "enabled": True,
            "trigger": trigger
        })
        
        executed_count = 0
        
        for rule in rules:
            if await self.evaluate_and_execute(rule, file, context):
                executed_count += 1
        
        return executed_count
    
    async def evaluate_and_execute(
        self,
        rule: Rule,
        file: FileRecord,
        context: dict = None
    ) -> bool:
        """
        Evaluate rule conditions and execute actions.
        
        Args:
            rule: Rule to evaluate
            file: FileRecord to evaluate
            context: Additional context
            
        Returns:
            True if rule was executed
        """
        try:
            # Evaluate all conditions
            if not await self._evaluate_conditions(rule, file, context):
                return False
            
            # Execute all actions
            success = await self._execute_actions(rule, file, context)
            
            if success:
                # Update statistics
                rule.execution_count += 1
                rule.last_executed = datetime.now().isoformat()
                await rule.save()
                
                logger.info(f"Executed rule: {rule.name} on {file.name}")
            
            return success
        
        except Exception as e:
            logger.error(f"Failed to execute rule {rule.name}: {e}")
            return False
    
    async def _evaluate_conditions(
        self,
        rule: Rule,
        file: FileRecord,
        context: dict
    ) -> bool:
        """
        Evaluate all conditions (AND logic).
        
        Args:
            rule: Rule with conditions
            file: FileRecord to evaluate
            context: Additional context
            
        Returns:
            True if all conditions match
        """
        for condition_def in rule.conditions:
            condition = create_condition(
                condition_def.get("type"),
                condition_def.get("params", {})
            )
            
            if not condition:
                logger.warning(f"Skipping invalid condition in rule {rule.name}")
                continue
            
            if not condition.evaluate(file, context):
                return False
        
        return True
    
    async def _execute_actions(
        self,
        rule: Rule,
        file: FileRecord,
        context: dict
    ) -> bool:
        """
        Execute all actions in order.
        
        Args:
            rule: Rule with actions
            file: FileRecord to act on
            context: Additional context
            
        Returns:
            True if all actions succeeded
        """
        for action_def in rule.actions:
            action = create_action(
                action_def.get("type"),
                action_def.get("params", {})
            )
            
            if not action:
                logger.warning(f"Skipping invalid action in rule {rule.name}")
                continue
            
            success = await action.execute(file, context)
            if not success:
                logger.warning(f"Action failed in rule {rule.name}")
                # Continue with other actions despite failure
        
        return True
    
    async def execute_manual(
        self,
        rule_id: ObjectId,
        file_ids: List[ObjectId]
    ) -> int:
        """
        Manually execute rule on specific files.
        
        Args:
            rule_id: Rule ObjectId
            file_ids: List of file ObjectIds
            
        Returns:
            Number of files processed
        """
        rule = await Rule.get(rule_id)
        if not rule:
            return 0
        
        processed = 0
        
        for file_id in file_ids:
            file = await FileRecord.get(file_id)
            if file:
                if await self.evaluate_and_execute(rule, file):
                    processed += 1
        
        logger.info(f"Manual execution: {processed}/{len(file_ids)} files")
        return processed
