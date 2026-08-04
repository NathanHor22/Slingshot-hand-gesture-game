# Hand Slingshot Game

This repository contains two clearly separated versions of the same game.

| Folder | Purpose |
| --- | --- |
| web/ | Browser version for GitHub Pages. |
| hand_slingshot_game/ | Windows Python and Pygame desktop prototype. |
| .github/workflows/ | Automatic GitHub Pages deployment. |

## Browser game and GitHub Pages

Push the repository to GitHub, then in the repository go to Settings, Pages,
and select GitHub Actions as the source. The included workflow publishes the
browser game automatically whenever you push to main. It publishes the web
folder.

The live game asks for webcam permission. If more than one camera is connected, a
Camera dropdown appears next to the buttons so you can pick which one to use. It
also has a mouse mode, so it can be played without a camera: select Try with
mouse, hold on the game to pull, and release to launch. Press R to restart.

For a local preview, serve the web folder with a static web server. Camera
access requires localhost or HTTPS.

## Desktop prototype

See hand_slingshot_game/README.md for Windows setup and run instructions.
