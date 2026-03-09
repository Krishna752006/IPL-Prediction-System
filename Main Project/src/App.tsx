import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Login from './pages/Login';
import Register from './pages/Register';
import Predictions from './pages/Predictions';
import Statistics from './pages/Statistics';
import TeamAnalysis from './pages/TeamAnalysis';
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <main className="container mx-auto px-4 py-8">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route 
              path="/predictions" 
              element={
                <ProtectedRoute>
                  <Predictions />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/statistics" 
              element={
                <ProtectedRoute requiresBusinessRole>
                  <Statistics />
                </ProtectedRoute>
              } 
            />
            <Route 
              path="/team-analysis" 
              element={
                <ProtectedRoute requiresBusinessRole>
                  <TeamAnalysis />
                </ProtectedRoute>
              } 
            />
          </Routes>
        </main>
      </div>
    </Router>
  );
}
export default App;