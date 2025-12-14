from src.naming import generate_name

def test_generate_name_unique():
    names = set()
    for _ in range(20):
        name = generate_name(used=names)
        assert name not in names
        names.add(name)
