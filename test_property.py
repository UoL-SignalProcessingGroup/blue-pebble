from stonesoup.base import Base, Property


class TestClass(Base):
    test_prop = Property(float, default=0.5, doc="Test property")


# Test instantiation
try:
    obj = TestClass()
    print("Success!")
    print(f"test_prop value: {obj.test_prop}")
except Exception as e:
    print(f"Error: {e}")
