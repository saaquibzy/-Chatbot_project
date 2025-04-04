knowledge_base = {
    "Python": {
        "Introduction": "Python is a powerful programming language used for AI, web development, and more.",
        "Data Types": "Python supports integers, floats, strings, lists, tuples, sets, and dictionaries.",
        "OOP": "Python supports object-oriented programming with classes, objects, and inheritance."
    },
    "C++": {
        "Introduction": "C++ is a high-performance compiled language used for game development and system programming.",
        "Inheritance": "C++ supports multiple types of inheritance including single, multiple, hierarchical, and hybrid.",
        "Polymorphism": "C++ allows function overloading and operator overloading as part of compile-time polymorphism.",
        "Memory Management": "C++ uses `new` and `delete` for dynamic memory allocation.",
        "Multithreading": "C++ supports multithreading using the `thread` library for parallel programming.",
        "Game Development": "C++ is used with game engines like Unreal Engine and Unity (via C++ integration)."
    },
    "Robotics": {
        "Arduino Basics": "Arduino is an open-source electronics platform for building robots and embedded systems.",
        "Motor Control": "Uses motor driver modules like L298N to control DC motors and servos.",
        "Obstacle Detection": "Uses ultrasonic and IR sensors for obstacle detection."
    }
}

# def get_knowledge(subject, topic):
#     if subject in knowledge_base and topic in knowledge_base[subject]:
#         return knowledge_base[subject][topic]
#     return "Sorry, I don't have information on that topic."

def get_knowledge(subject, topic):
    subject_data = knowledge_base.get(subject)
    if not subject_data:
        return "I don't have knowledge on that subject."

    closest_match = next((key for key in subject_data if key.lower() in topic.lower()), None)
    return subject_data[closest_match] if closest_match else "I don't have information on that topic."