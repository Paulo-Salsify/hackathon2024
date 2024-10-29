from pygame import mixer

def load_sound(sound_path: str) -> mixer.Sound:
    mixer.init()
    sound = mixer.Sound(sound_path)
    return sound

def play_sound(sound: mixer.Sound) -> None:
    sound.play()

def stop_sound(sound: mixer.Sound) -> None:
    sound.stop()
