import sys
import os

# Ensure src is in path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from translator import NeoRomanceTranslator

translator = NeoRomanceTranslator()
lorem_text = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. 
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
"""

print("--- Tradução Neo-Românica do Lorem Ipsum ---")
print(translator.translate_text(lorem_text))
