"""
Test suite for build function with dataclass and Pydantic model support.
Tests model instantiation, type validation, error handling, and nested models.
"""
import pytest
import asyncio
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, ValidationError, validator

# Import the enhanced build function and other operations
from fp_ops import operation, Operation
from fp_ops.objects import get, build, merge


# Test Models
class PydanticUserV2(BaseModel):
    """Pydantic v2 model for testing."""
    id: int
    name: str
    email: str
    is_active: bool = True
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v


class PydanticUserV1(BaseModel):
    """Pydantic v1 style model for testing."""
    id: int
    name: str
    email: str
    is_active: bool = True
    
    class Config:
        validate_assignment = True


@dataclass
class DataclassUser:
    """Dataclass model for testing."""
    id: int
    name: str
    email: str
    is_active: bool = True


@dataclass
class DataclassProduct:
    """Product dataclass with nested fields."""
    name: str
    price: float
    category: str = "general"
    tags: List[str] = field(default_factory=list)


class RegularUser:
    """Regular Python class for testing."""
    def __init__(self, id: int, name: str, email: str, is_active: bool = True):
        self.id = id
        self.name = name
        self.email = email
        self.is_active = is_active


# Nested models for complex testing
@dataclass
class Address:
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None


@dataclass
class DataclassUserWithAddress:
    id: int
    name: str
    email: str
    address: Address
    is_active: bool = True


class PydanticAddress(BaseModel):
    street: str
    city: str
    country: str
    postal_code: Optional[str] = None


class PydanticUserWithAddress(BaseModel):
    id: int
    name: str
    email: str
    address: PydanticAddress
    is_active: bool = True


# Test fixtures
@pytest.fixture
def user_data():
    return {
        "user_id": 123,
        "full_name": "John Doe",
        "contact_email": "john@example.com",
        "status": {"active": True}
    }


@pytest.fixture
def product_data():
    return {
        "title": "Premium Widget",
        "cost": "99.99",
        "product_category": "electronics",
        "keywords": ["tech", "gadget", "premium"]
    }


@pytest.fixture
def nested_user_data():
    return {
        "id": 456,
        "user": {
            "first_name": "Jane",
            "last_name": "Smith",
            "contact": {
                "email": "jane@example.com",
                "phone": "555-0123"
            }
        },
        "location": {
            "street_address": "123 Main St",
            "city_name": "New York",
            "country_code": "US",
            "zip": "10001"
        },
        "active_status": True
    }


class TestBuildWithModels:
    """Test suite for build function with model support."""
    
    @pytest.mark.asyncio
    async def test_build_with_pydantic_v2(self, user_data):
        """Test building a Pydantic v2 model."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        build_op = build(schema, PydanticUserV2)
        result = await build_op.execute(user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, PydanticUserV2)
        assert user.id == 123
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_build_with_pydantic_v1(self, user_data):
        """Test building a Pydantic v1 style model."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        build_op = build(schema, PydanticUserV1)
        result = await build_op.execute(user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, PydanticUserV1)
        assert user.id == 123
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_build_with_dataclass(self, user_data):
        """Test building a dataclass instance."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        build_op = build(schema, DataclassUser)
        result = await build_op.execute(user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, DataclassUser)
        assert user.id == 123
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_build_with_regular_class(self, user_data):
        """Test building a regular class instance."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        build_op = build(schema, RegularUser)
        result = await build_op.execute(user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, RegularUser)
        assert user.id == 123
        assert user.name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_active is True
    
    @pytest.mark.asyncio
    async def test_build_with_type_conversions(self, product_data):
        """Test build with type conversions in operations."""
        schema = {
            "name": get("title"),
            "price": get("cost") >> operation(lambda x: float(x)),
            "category": get("product_category"),
            "tags": get("keywords")
        }
        
        build_op = build(schema, DataclassProduct)
        result = await build_op.execute(product_data)
        
        assert result.is_ok()
        product = result.default_value(None)
        assert isinstance(product, DataclassProduct)
        assert product.name == "Premium Widget"
        assert product.price == 99.99
        assert product.category == "electronics"
        assert product.tags == ["tech", "gadget", "premium"]
    
    @pytest.mark.asyncio
    async def test_build_with_defaults(self):
        """Test build with model defaults when fields are missing."""
        minimal_data = {
            "user_id": 789,
            "full_name": "Alice Brown",
            "contact_email": "alice@example.com"
        }
        
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            # is_active not provided, should use model default
        }
        
        build_op = build(schema, DataclassUser)
        result = await build_op.execute(minimal_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert user.id == 789
        assert user.name == "Alice Brown"
        assert user.email == "alice@example.com"
        assert user.is_active is True  # Default value
    
    @pytest.mark.asyncio
    async def test_build_with_validation_error(self, user_data):
        """Test build handles validation errors from Pydantic."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": lambda d: "invalid-email",  # No @ sign
            "is_active": get("status.active")
        }
        
        build_op = build(schema, PydanticUserV2)
        result = await build_op.execute(user_data)
        
        # Should fail due to validation error
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        assert "Failed to instantiate" in str(result.error)
    
    @pytest.mark.asyncio
    async def test_build_with_missing_required_field(self):
        """Test build fails when required fields are missing."""
        incomplete_data = {
            "user_id": 999,
            # Missing name and email
        }
        
        schema = {
            "id": get("user_id"),
            # name and email will be None
        }
        
        build_op = build(schema, DataclassUser)
        result = await build_op.execute(incomplete_data)
        
        # Should fail because required fields are missing
        assert result.is_error()
        assert isinstance(result.error, ValueError)
    
    @pytest.mark.asyncio
    async def test_build_with_nested_dataclass(self, nested_user_data):
        """Test building models with nested dataclasses."""
        # First build the address
        address_schema = {
            "street": get("location.street_address"),
            "city": get("location.city_name"),
            "country": get("location.country_code"),
            "postal_code": get("location.zip")
        }
        
        # Then build the user with the address
        user_schema = {
            "id": get("id"),
            "name": lambda d: f"{d['user']['first_name']} {d['user']['last_name']}",
            "email": get("user.contact.email"),
            "address": build(address_schema, Address),
            "is_active": get("active_status")
        }
        
        build_op = build(user_schema, DataclassUserWithAddress)
        result = await build_op.execute(nested_user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, DataclassUserWithAddress)
        assert user.id == 456
        assert user.name == "Jane Smith"
        assert user.email == "jane@example.com"
        assert isinstance(user.address, Address)
        assert user.address.street == "123 Main St"
        assert user.address.city == "New York"
        assert user.address.country == "US"
        assert user.address.postal_code == "10001"
    
    @pytest.mark.asyncio
    async def test_build_with_nested_pydantic(self, nested_user_data):
        """Test building models with nested Pydantic models."""
        address_schema = {
            "street": get("location.street_address"),
            "city": get("location.city_name"),
            "country": get("location.country_code"),
            "postal_code": get("location.zip")
        }
        
        user_schema = {
            "id": get("id"),
            "name": lambda d: f"{d['user']['first_name']} {d['user']['last_name']}",
            "email": get("user.contact.email"),
            "address": build(address_schema, PydanticAddress),
            "is_active": get("active_status")
        }
        
        build_op = build(user_schema, PydanticUserWithAddress)
        result = await build_op.execute(nested_user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, PydanticUserWithAddress)
        assert user.id == 456
        assert user.name == "Jane Smith"
        assert user.email == "jane@example.com"
        assert isinstance(user.address, PydanticAddress)
        assert user.address.street == "123 Main St"
        assert user.address.city == "New York"
        assert user.address.country == "US"
        assert user.address.postal_code == "10001"
    
    @pytest.mark.asyncio
    async def test_build_backward_compatibility(self, user_data):
        """Test that build without model parameter still returns dict."""
        schema = {
            "id": get("user_id"),
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        # No model parameter - should return dict
        build_op = build(schema)
        result = await build_op.execute(user_data)
        
        assert result.is_ok()
        output = result.default_value(None)
        assert isinstance(output, dict)
        assert output["id"] == 123
        assert output["name"] == "John Doe"
        assert output["email"] == "john@example.com"
        assert output["is_active"] is True
    
    @pytest.mark.asyncio
    async def test_build_with_complex_transformations(self, nested_user_data):
        """Test build with complex data transformations before model creation."""
        @operation
        def format_phone(phone: str) -> str:
            # Remove non-digits and format
            digits = ''.join(c for c in phone if c.isdigit())
            return f"+1-{digits[:3]}-{digits[3:6]}-{digits[6:]}"
        
        @operation
        def calculate_user_score(data: Dict) -> int:
            # Dummy scoring logic
            base_score = 100
            if data.get("active_status"):
                base_score += 50
            if "@" in data.get("user", {}).get("contact", {}).get("email", ""):
                base_score += 25
            return base_score
        
        # Custom model with score
        @dataclass
        class ScoredUser:
            id: int
            name: str
            email: str
            phone: str
            score: int
            status: str
        
        schema = {
            "id": get("id"),
            "name": lambda d: f"{d['user']['first_name']} {d['user']['last_name']}",
            "email": get("user.contact.email"),
            "phone": get("user.contact.phone") >> format_phone,
            "score": calculate_user_score,
            "status": lambda d: "active" if d.get("active_status") else "inactive"
        }
        
        build_op = build(schema, ScoredUser)
        result = await build_op.execute(nested_user_data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, ScoredUser)
        assert user.id == 456
        assert user.name == "Jane Smith"
        assert user.email == "jane@example.com"
        assert user.phone == "+1-555-012-3"
        assert user.score == 175  # 100 base + 50 active + 25 email
        assert user.status == "active"
    
    @pytest.mark.asyncio
    async def test_build_in_pipeline(self, product_data):
        """Test using build with models in a pipeline."""
        # First transform the data
        transform_op = build({
            "name": get("title"),
            "price": get("cost") >> operation(lambda x: float(x)),
            "category": get("product_category"),
            "tags": get("keywords"),
            "metadata": {
                "source": "api",
                "processed": True
            }
        })
        
        # Then create the model
        model_op = build({
            "name": get("name"),
            "price": get("price"),
            "category": get("category"),
            "tags": get("tags")
        }, DataclassProduct)
        
        # Compose the pipeline
        pipeline = transform_op >> model_op
        
        result = await pipeline.execute(product_data)
        assert result.is_ok()
        product = result.default_value(None)
        assert isinstance(product, DataclassProduct)
        assert product.name == "Premium Widget"
        assert product.price == 99.99
        assert product.category == "electronics"
        assert product.tags == ["tech", "gadget", "premium"]
    
    @pytest.mark.asyncio
    async def test_build_with_type_mismatch(self, user_data):
        """Test build handles type mismatches gracefully."""
        schema = {
            "id": lambda d: "not-a-number",  # String instead of int
            "name": get("full_name"),
            "email": get("contact_email"),
            "is_active": get("status.active")
        }
        
        # Test with Pydantic model which validates types
        build_op = build(schema, PydanticUserV2)
        result = await build_op.execute(user_data)
        
        # Should fail due to type mismatch in Pydantic validation
        assert result.is_error()
        assert isinstance(result.error, ValueError)
        
        # Test with dataclass - it accepts the wrong type (no runtime validation)
        build_op_dc = build(schema, DataclassUser)
        result_dc = await build_op_dc.execute(user_data)
        
        # Dataclasses don't validate types at runtime, so this succeeds
        assert result_dc.is_ok()
        user = result_dc.default_value(None)
        assert user.id == "not-a-number"  # Wrong type but accepted
    
    @pytest.mark.asyncio
    async def test_build_with_optional_fields(self):
        """Test build with optional fields in models."""
        @dataclass
        class OptionalFieldsModel:
            required_field: str
            optional_str: Optional[str] = None
            optional_int: Optional[int] = None
            optional_list: Optional[List[str]] = None
        
        data = {
            "req": "required value",
            "opt_str": "optional value",
            # opt_int and opt_list are missing
        }
        
        schema = {
            "required_field": get("req"),
            "optional_str": get("opt_str"),
            "optional_int": get("opt_int"),  # Will be None
            "optional_list": get("opt_list")  # Will be None
        }
        
        build_op = build(schema, OptionalFieldsModel)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        model = result.default_value(None)
        assert model.required_field == "required value"
        assert model.optional_str == "optional value"
        assert model.optional_int is None
        assert model.optional_list is None
    
    @pytest.mark.asyncio
    async def test_build_with_list_of_models(self):
        """Test building a model containing a list of other models."""
        @dataclass
        class Item:
            id: int
            name: str
            price: float
        
        @dataclass
        class Order:
            order_id: str
            items: List[Item]
            total: float
        
        order_data = {
            "id": "ORD-123",
            "line_items": [
                {"item_id": 1, "product_name": "Widget A", "cost": 10.0},
                {"item_id": 2, "product_name": "Widget B", "cost": 20.0},
                {"item_id": 3, "product_name": "Widget C", "cost": 15.0}
            ]
        }
        
        # Build items using a list comprehension in lambda
        schema = {
            "order_id": get("id"),
            "items": lambda d: [
                Item(
                    id=item["item_id"],
                    name=item["product_name"],
                    price=item["cost"]
                )
                for item in d["line_items"]
            ],
            "total": lambda d: sum(item["cost"] for item in d["line_items"])
        }
        
        build_op = build(schema, Order)
        result = await build_op.execute(order_data)
        
        assert result.is_ok()
        order = result.default_value(None)
        assert isinstance(order, Order)
        assert order.order_id == "ORD-123"
        assert len(order.items) == 3
        assert all(isinstance(item, Item) for item in order.items)
        assert order.items[0].name == "Widget A"
        assert order.items[1].name == "Widget B"
        assert order.items[2].name == "Widget C"
        assert order.total == 45.0


class TestBuildModelEdgeCases:
    """Test edge cases for build with models."""
    
    @pytest.mark.asyncio
    async def test_build_with_inheritance(self):
        """Test build with model inheritance."""
        class BaseUser(BaseModel):
            id: int
            name: str
        
        class ExtendedUser(BaseUser):
            email: str
            role: str = "user"
        
        data = {
            "user_id": 123,
            "username": "johndoe",
            "user_email": "john@example.com",
            "user_role": "admin"
        }
        
        schema = {
            "id": get("user_id"),
            "name": get("username"),
            "email": get("user_email"),
            "role": get("user_role")
        }
        
        build_op = build(schema, ExtendedUser)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, ExtendedUser)
        assert isinstance(user, BaseUser)  # Also instance of parent
        assert user.id == 123
        assert user.name == "johndoe"
        assert user.email == "john@example.com"
        assert user.role == "admin"
    
    @pytest.mark.asyncio
    async def test_build_with_custom_init(self):
        """Test build with class that has custom __init__ logic."""
        class CustomInitClass:
            def __init__(self, value: int, multiplier: int = 2):
                self.value = value
                self.result = value * multiplier
                self.timestamp = "2024-01-01"  # Set internally
        
        data = {"base_value": 10, "mult": 3}
        
        schema = {
            "value": get("base_value"),
            "multiplier": get("mult")
        }
        
        build_op = build(schema, CustomInitClass)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        obj = result.default_value(None)
        assert isinstance(obj, CustomInitClass)
        assert obj.value == 10
        assert obj.result == 30  # 10 * 3
        assert obj.timestamp == "2024-01-01"
    
    @pytest.mark.asyncio
    async def test_build_with_property_decorator(self):
        """Test build with models using property decorators."""
        class PropertyUser:
            def __init__(self, first_name: str, last_name: str, age: int):
                self.first_name = first_name
                self.last_name = last_name
                self.age = age
            
            @property
            def full_name(self):
                return f"{self.first_name} {self.last_name}"
            
            @property
            def is_adult(self):
                return self.age >= 18
        
        data = {
            "fname": "John",
            "lname": "Doe",
            "user_age": 25
        }
        
        schema = {
            "first_name": get("fname"),
            "last_name": get("lname"),
            "age": get("user_age")
        }
        
        build_op = build(schema, PropertyUser)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        user = result.default_value(None)
        assert isinstance(user, PropertyUser)
        assert user.full_name == "John Doe"  # Property works
        assert user.is_adult is True  # Property works
    
    @pytest.mark.asyncio
    async def test_build_with_slots(self):
        """Test build with classes using __slots__."""
        class SlottedClass:
            __slots__ = ['id', 'name', 'value']
            
            def __init__(self, id: int, name: str, value: float):
                self.id = id
                self.name = name
                self.value = value
        
        data = {"id": 1, "name": "test", "value": 3.14}
        
        build_op = build(data, SlottedClass)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        obj = result.default_value(None)
        assert isinstance(obj, SlottedClass)
        assert obj.id == 1
        assert obj.name == "test"
        assert obj.value == 3.14
    
    @pytest.mark.asyncio
    async def test_build_model_with_post_init(self):
        """Test build with dataclass using __post_init__."""
        @dataclass
        class PostInitModel:
            raw_price: str
            quantity: int
            price: float = field(init=False)
            total: float = field(init=False)
            
            def __post_init__(self):
                self.price = float(self.raw_price.replace("$", ""))
                self.total = self.price * self.quantity
        
        data = {
            "price_str": "$19.99",
            "qty": 3
        }
        
        schema = {
            "raw_price": get("price_str"),
            "quantity": get("qty")
        }
        
        build_op = build(schema, PostInitModel)
        result = await build_op.execute(data)
        
        assert result.is_ok()
        model = result.default_value(None)
        assert model.raw_price == "$19.99"
        assert model.quantity == 3
        assert model.price == 19.99  # Processed in __post_init__
        assert model.total == 59.97  # Calculated in __post_init__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])