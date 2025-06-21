'use client'

import { useState, useEffect } from 'react'
import io from 'socket.io-client'
import GestureDisplay from '@/components/GestureDisplay'
import ConnectionStatus from '@/components/ConnectionStatus'

export default function Home() {
  const [gesture, setGesture] = useState<string>('No gesture detected')
  const [confidence, setConfidence] = useState<number>(0)
  const [isConnected, setIsConnected] = useState<boolean>(false)
  const [socket, setSocket] = useState<any>(null)

  useEffect(() => {
    // Connect to the Python backend
    const newSocket = io('http://localhost:5000')
    
    newSocket.on('connect', () => {
      console.log('Connected to backend')
      setIsConnected(true)
    })

    newSocket.on('disconnect', () => {
      console.log('Disconnected from backend')
      setIsConnected(false)
    })

    newSocket.on('gesture_update', (data: { gesture: string; confidence: number }) => {
      setGesture(data.gesture)
      setConfidence(data.confidence)
    })

    setSocket(newSocket)

    return () => {
      newSocket.close()
    }
  }, [])

  return (
    <main className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-4">
            Sign Language Recognition
          </h1>
          <p className="text-lg text-gray-600">
            Real-time hand gesture detection and classification
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              Current Gesture
            </h2>
            <GestureDisplay gesture={gesture} confidence={confidence} />
          </div>

          <div className="bg-white rounded-lg shadow-lg p-6">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">
              System Status
            </h2>
            <ConnectionStatus isConnected={isConnected} />
            
            <div className="mt-6">
              <h3 className="text-lg font-medium text-gray-700 mb-3">
                Supported Gestures
              </h3>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="bg-green-100 p-2 rounded">Hello</div>
                <div className="bg-green-100 p-2 rounded">Thanks</div>
                <div className="bg-green-100 p-2 rounded">Yes</div>
                <div className="bg-green-100 p-2 rounded">No</div>
                <div className="bg-green-100 p-2 rounded">I Love You</div>
                <div className="bg-green-100 p-2 rounded">Sad</div>
                <div className="bg-green-100 p-2 rounded">Happy</div>
              </div>
            </div>
          </div>
        </div>

        <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-semibold text-gray-800 mb-4">
            Instructions
          </h2>
          <div className="text-gray-600 space-y-2">
            <p>• Make sure your camera is connected and accessible</p>
            <p>• Position your hand clearly in front of the camera</p>
            <p>• Perform one of the supported sign language gestures</p>
            <p>• The detected gesture will appear in real-time</p>
            <p>• Ensure good lighting for better detection accuracy</p>
          </div>
        </div>
      </div>
    </main>
  )
} 