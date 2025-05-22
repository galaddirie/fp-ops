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
        
        result = self.model_copy(deep=True)
        
        # Create a deep copy of other's data to avoid modifying it
        other_data = other.model_dump()
        
        for field_name, field_value in other_data.items():
            if field_name == 'metadata':
                # Create a new metadata dict instead of updating in place
                result.metadata = {**result.metadata, **field_value}
            else:
                current_value = getattr(result, field_name, None)
                if (isinstance(current_value, BaseContext) and 
                    isinstance(field_value, dict)):
                    field_cls = type(current_value)
                    new_context = field_cls(**field_value)
                    setattr(result, field_name, current_value.merge(new_context))
                elif (isinstance(current_value, BaseContext) and 
                    isinstance(field_value, BaseContext)):
                    setattr(result, field_name, current_value.merge(field_value))
                else:
                    setattr(result, field_name, field_value)
        
        return result
