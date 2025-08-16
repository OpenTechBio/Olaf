from abc import ABC, abstractmethod
import json

class AutoMetric(ABC):
    """
    Abstract base class for a metric to be applied to an AnnData object.
    """ 
    @abstractmethod
    def metric(self, adata) -> dict:
        """
        Run the metric and return a dictionary of results.
        """
        pass
    
    def run(self, adata):
        """
        Handles execution + JSON serialization.
        """
        result = self.metric(adata)
        print(json.dumps(result))  # Always print result at the end