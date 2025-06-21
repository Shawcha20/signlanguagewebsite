# Sign Language Recognition Frontend

A Next.js application that displays real-time hand gesture recognition results from the Python backend.

## Features

- **Real-time Gesture Display**: Shows detected sign language gestures in real-time
- **Confidence Visualization**: Displays confidence levels with color-coded progress bars
- **Connection Status**: Shows connection status to the Python backend
- **Responsive Design**: Works on desktop and mobile devices
- **WebSocket Communication**: Real-time updates via WebSocket connection

## Prerequisites

Before running this frontend, make sure you have:

1. **Python Backend Running**: The Python sign language detection backend must be running
2. **Trained Model**: The `sign_language_model.h5` file should exist in the backend directory
3. **Camera Access**: A webcam should be connected and accessible

## Installation

1. **Install Dependencies**:
   ```bash
   npm install
   ```

2. **Start the Development Server**:
   ```bash
   npm run dev
   ```

3. **Open the Application**:
   Navigate to [http://localhost:3000](http://localhost:3000) in your browser

## Backend Setup

Before using the frontend, you need to start the Python backend:

1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the Backend Server**:
   ```bash
   python backend_server.py
   ```

   This will:
   - Train the model if it doesn't exist
   - Start the WebSocket server on port 5000
   - Begin real-time gesture detection

## Usage

1. **Start the Backend**: Run `python backend_server.py` in the main project directory
2. **Start the Frontend**: Run `npm run dev` in the `frontend` directory
3. **Open the Application**: Go to `http://localhost:3000`
4. **Perform Gestures**: Show sign language gestures to your camera
5. **View Results**: The detected gesture and confidence will appear in real-time

## Supported Gestures

The system recognizes the following sign language gestures:
- **Hello**: Greeting gesture
- **Thanks**: Thank you gesture
- **Yes**: Affirmative gesture
- **No**: Negative gesture
- **I Love You**: Combined gesture
- **Sad**: Sadness expression
- **Happy**: Happiness expression

## Troubleshooting

### Connection Issues
- Make sure the Python backend is running on port 5000
- Check that your firewall allows connections to localhost:5000
- Ensure the backend server started without errors

### Camera Issues
- Make sure your camera is connected and not used by another application
- Check camera permissions in your browser
- Try refreshing the page if the camera doesn't work

### Model Issues
- Ensure the `sign_language_model.h5` file exists in the backend directory
- If the model doesn't exist, the backend will automatically train a new one
- Check that you have sufficient training data in the `train/` and `test/` directories

## Development

### Project Structure
```
frontend/
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
│   └── components/
│       ├── GestureDisplay.tsx
│       └── ConnectionStatus.tsx
├── package.json
├── tailwind.config.ts
└── tsconfig.json
```

### Key Components

- **`page.tsx`**: Main application page with WebSocket connection
- **`GestureDisplay.tsx`**: Displays current gesture and confidence
- **`ConnectionStatus.tsx`**: Shows backend connection status

### WebSocket Events

The frontend listens for these WebSocket events from the backend:
- `gesture_update`: Contains current gesture and confidence
- `status`: Connection status messages
- `error`: Error messages from the backend

## Build for Production

To build the application for production:

```bash
npm run build
npm start
```

## Technologies Used

- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework
- **Socket.IO Client**: WebSocket communication
- **React Hooks**: State management and side effects 