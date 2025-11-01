def detect_intent(message):
    message = message.lower()
    if "crop" in message or "grow" in message:
        return "crop_recommendation"
    elif "fertilizer" in message:
        return "fertilizer_suggestion"
    elif "disease" in message or "pest" in message:
        return "disease_detection"
    elif "weather" in message:
        return "weather_query"
    elif "price" in message or "market" in message:
        return "market_price_query"
    else:
        return "general"
