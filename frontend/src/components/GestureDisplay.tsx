interface GestureDisplayProps {
  gesture: string
  confidence: number
}

export default function GestureDisplay({ gesture, confidence }: GestureDisplayProps) {
  const getGestureColor = (gesture: string) => {
    switch (gesture.toLowerCase()) {
      case 'hello':
        return 'text-blue-600'
      case 'thanks':
        return 'text-green-600'
      case 'yes':
        return 'text-emerald-600'
      case 'no':
        return 'text-red-600'
      case 'iloveu':
        return 'text-pink-600'
      case 'sad':
        return 'text-gray-600'
      case 'happy':
        return 'text-yellow-600'
      default:
        return 'text-gray-500'
    }
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return 'text-green-600'
    if (confidence >= 0.6) return 'text-yellow-600'
    if (confidence >= 0.4) return 'text-orange-600'
    return 'text-red-600'
  }

  const getConfidenceText = (confidence: number) => {
    if (confidence >= 0.8) return 'High'
    if (confidence >= 0.6) return 'Medium'
    if (confidence >= 0.4) return 'Low'
    return 'Very Low'
  }

  return (
    <div className="space-y-6">
      <div className="text-center">
        <div className="text-6xl font-bold mb-4">
          {gesture === 'No gesture detected' ? '🤔' : '✋'}
        </div>
        <h3 className={`text-3xl font-bold ${getGestureColor(gesture)}`}>
          {gesture}
        </h3>
      </div>

      {gesture !== 'No gesture detected' && (
        <div className="space-y-4">
          <div className="text-center">
            <p className="text-sm text-gray-600 mb-2">Confidence Level</p>
            <div className="flex items-center justify-center space-x-2">
              <div className="w-32 bg-gray-200 rounded-full h-3">
                <div
                  className="bg-gradient-to-r from-red-500 via-yellow-500 to-green-500 h-3 rounded-full transition-all duration-300"
                  style={{ width: `${confidence * 100}%` }}
                ></div>
              </div>
              <span className={`text-sm font-semibold ${getConfidenceColor(confidence)}`}>
                {Math.round(confidence * 100)}%
              </span>
            </div>
            <p className={`text-sm font-medium mt-1 ${getConfidenceColor(confidence)}`}>
              {getConfidenceText(confidence)} Confidence
            </p>
          </div>
        </div>
      )}

      {gesture === 'No gesture detected' && (
        <div className="text-center text-gray-500">
          <p>Position your hand in front of the camera</p>
          <p className="text-sm mt-2">Make sure you have good lighting</p>
        </div>
      )}
    </div>
  )
} 