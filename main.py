import time
from voice_module import recognize_speech, speak
from sarcasm_model import detect_sarcasm
from sentiment_model import analyze_sentiment
from knowledge_base import get_knowledge

def chatbot_response(user_input):
    sarcasm_detected = detect_sarcasm(user_input)
    sentiment, score = analyze_sentiment(user_input)

    if sarcasm_detected:
        return "Oh, sarcasm detected! 😂 But let’s be real for a second..."
    
    if sentiment == "POSITIVE":
        return "That's great to hear! 😊"
    elif sentiment == "NEGATIVE":
        return "I'm sorry to hear that. 😞 I hope things get better."
    else:
        return "That sounds interesting! Tell me more. 🤔"

def chatbot():
    speak("\nWelcome to the Smart Chatbot! Say 'exit' to quit.\n")

    while True:
        user_input = recognize_speech()
        if user_input is None:
            continue
        if user_input.lower() == "exit":
            speak("Goodbye! 👋")
            break

        parts = user_input.split(" ")
        if len(parts) >= 2 and parts[0].lower() in ["python", "c++", "robotics"]:
            subject = parts[0].capitalize()
            topic = " ".join(parts[1:])
            response = get_knowledge(subject, topic)
        else:
            response = chatbot_response(user_input)

        speak(response)

if __name__ == "__main__":
    chatbot()
