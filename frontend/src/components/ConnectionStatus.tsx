interface ConnectionStatusProps {
  isConnected: boolean
}

export default function ConnectionStatus({ isConnected }: ConnectionStatusProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center space-x-3">
        <div
          className={`w-3 h-3 rounded-full ${
            isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
          }`}
        ></div>
        <span className={`font-medium ${isConnected ? 'text-green-600' : 'text-red-600'}`}>
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>

      <div className="text-sm text-gray-600">
        {isConnected ? (
          <div className="space-y-2">
            <p>✅ Backend server is running</p>
            <p>✅ Camera feed is active</p>
            <p>✅ Gesture detection is ready</p>
          </div>
        ) : (
          <div className="space-y-2">
            <p>❌ Backend server not found</p>
            <p>❌ Camera feed unavailable</p>
            <p>❌ Gesture detection paused</p>
          </div>
        )}
      </div>

      {!isConnected && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h4 className="font-medium text-yellow-800 mb-2">Troubleshooting</h4>
          <ul className="text-sm text-yellow-700 space-y-1">
            <li>• Make sure the Python backend is running</li>
            <li>• Check if the server is on port 5000</li>
            <li>• Ensure your camera is connected</li>
            <li>• Try refreshing the page</li>
          </ul>
        </div>
      )}
    </div>
  )
} 