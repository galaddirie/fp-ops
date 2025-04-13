from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, Type, TypeVar, Generic

class BaseContext(BaseModel):
    """Base class for all operation contexts."""
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class ConfigDict:
        arbitrary_types_allowed = True
    
    def merge(self, other: 'BaseContext') -> 'BaseContext':
        """
        Merge another context into this one.
        
        Args:
            other: Another context to merge with this one
            
        Returns:
            A new context with the combined properties
        """
        if not isinstance(other, BaseContext):
            raise TypeError(f"Cannot merge {type(other)} with {type(self)}")
        
        # Create a copy of this context
        result = self.model_copy()
        
        # Get all fields from both contexts
        for field_name, field_value in other.model_dump().items():
            if field_name == 'metadata':
                # Special handling for metadata dictionary
                result.metadata.update(field_value)
            else:
                # For nested contexts, try to merge them recursively
                current_value = getattr(result, field_name, None)
                if (isinstance(current_value, BaseContext) and 
                    isinstance(field_value, dict)):
                    # Create a new context from the dict and merge
                    field_cls = type(current_value)
                    new_context = field_cls(**field_value)
                    setattr(result, field_name, current_value.merge(new_context))
                elif (isinstance(current_value, BaseContext) and 
                    isinstance(field_value, BaseContext)):
                    # Merge the two context objects
                    setattr(result, field_name, current_value.merge(field_value))
                else:
                    # For regular fields, overwrite with the other value
                    setattr(result, field_name, field_value)
        
        return result
