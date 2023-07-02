import random

# TODO: revise usefulness of these functions
def set_random_number_generator(object, random_number_generator: random.Random = None, seed: int = 0):
    if random_number_generator is None:
        random_number_generator = random.Random()
    if seed is not None:
        random_number_generator.seed(seed)
    object.random_number_generator = random_number_generator

def generate_random_number_generator(seed: int = 0):
    random_number_generator = random.Random()
    random_number_generator.seed(seed)
    return random_number_generator