"""
A simple example Python library module.

This module demonstrates basic Python functionality including
functions, classes, and constants.
"""

# Module-level constant
DEFAULT_GREETING = "Hello"


def greet(name: str, greeting: str = DEFAULT_GREETING) -> str:
    """
    Generate a greeting message.
    
    Args:
        name: The name of the person to greet
        greeting: The greeting to use (default: "Hello")
    
    Returns:
        A formatted greeting string
    
    Examples:
        >>> greet("Alice")
        'Hello, Alice!'
        >>> greet("Bob", "Hi")
        'Hi, Bob!'
    """
    return f"{greeting}, {name}!"


def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number
        b: Second number
    
    Returns:
        The sum of a and b
    """
    return a + b


class Calculator:
    """A simple calculator class."""
    
    def __init__(self, initial_value: float = 0):
        """
        Initialize the calculator.
        
        Args:
            initial_value: Starting value for the calculator
        """
        self.value = initial_value
    
    def add(self, amount: float) -> float:
        """Add amount to the current value."""
        self.value += amount
        return self.value
    
    def subtract(self, amount: float) -> float:
        """Subtract amount from the current value."""
        self.value -= amount
        return self.value
    
    def multiply(self, factor: float) -> float:
        """Multiply the current value by factor."""
        self.value *= factor
        return self.value
    
    def reset(self) -> None:
        """Reset the calculator to zero."""
        self.value = 0
    
    def __str__(self) -> str:
        """Return string representation of the calculator."""
        return f"Calculator(value={self.value})"


if __name__ == "__main__":
    # Example usage
    print(greet("World"))
    print(f"2 + 3 = {add_numbers(2, 3)}")
    
    calc = Calculator(10)
    print(calc)
    calc.add(5)
    print(f"After adding 5: {calc}")
