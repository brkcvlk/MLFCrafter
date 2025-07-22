import logging

# Setup logger for FlowChain
logger = logging.getLogger("flowcraft.FlowChain")

class FlowChain:
    """
    Initialize FlowChain with multiple crafters
    Usage: FlowChain(DataIngestCrafter(...), CleanerCrafter(...), ...)
    """
    def __init__(self, *crafters):
        self.crafters = list(crafters)
        logger.info(f"FlowChain initialized with {len(self.crafters)} crafters")
        for i, crafter in enumerate(self.crafters, 1):
            logger.debug(f"Crafter {i}: {type(crafter).__name__}")

    def add_crafter(self, crafter):
        """Add a single crafter to the chain"""
        self.crafters.append(crafter)
        logger.info(f"Added {type(crafter).__name__} to chain")

    def run(self, target_column=None, **kwargs):
        """
        Run the entire pipeline chain
        Args:
            target_column: Target column name for ML tasks
            **kwargs: Additional parameters to pass to the first crafter
        """
        logger.info("="*50)
        logger.info("STARTING FLOWCRAFT PIPELINE")
        logger.info("="*50)
        
        if target_column:
            logger.info(f"Target column: {target_column}")
        
        # Initialize context
        context = {
            "data": None,
            "target_column": target_column,
            **kwargs
        }

        # Execute each crafter in sequence
        for i, crafter in enumerate(self.crafters, 1):
            crafter_name = type(crafter).__name__
            logger.info(f"[{i}/{len(self.crafters)}] Running {crafter_name}...")
            
            try:
                context = crafter.run(context)
                if not isinstance(context, dict):
                    raise TypeError(f"Crafter {i} ({crafter_name}) must return a dict (context).")
                logger.info(f"[{i}/{len(self.crafters)}] {crafter_name} completed successfully")
                
            except Exception as e:
                logger.error(f"[{i}/{len(self.crafters)}] {crafter_name} failed: {str(e)}")
                raise RuntimeError(f"Error in crafter {i} ({crafter_name}): {str(e)}")

        logger.info("="*50)
        logger.info("FLOWCRAFT PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        
        return context
