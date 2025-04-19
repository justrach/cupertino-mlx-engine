# Remove FastAPI import
# from fastapi import APIRouter

# Import TurboAPI app instances from submodules
# Remove chat_app import as its routes will be defined in main.py
# from .chat.router import chat_app
# from .chat.models import models # Assuming models routes are part of chat_app or another app
# from .images import images # Assuming images routes are in images_app
# from .stt import stt as stt_router # Assuming stt routes are in stt_app
# from .tts import tts as tts_router # Assuming tts routes are in tts_app

# Placeholder imports for other modules (assuming they follow the same pattern)
# Replace these with actual imports when other modules are converted
# from .stt.router import stt_app
# from .tts.router import tts_app
# from .images.router import images_app
# from .models.router import models_app # If models has its own routes

# Collect the sub-apps to be mounted by main.py
# We'll use a dictionary mapping the desired mount prefix to the app instance,
# as TurboAPI requires mounting for full sub-applications.
sub_apps = {
    # "/stt": stt_app, # Uncomment when stt_app is available
    # "/tts": tts_app, # Uncomment when tts_app is available
    # "/models": models_app, # Uncomment when models_app is available
    # "/images": images_app, # Uncomment when images_app is available
    # "/chat": chat_app, # Removed: Chat routes defined directly in main.py
}
