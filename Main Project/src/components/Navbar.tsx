import React from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Trophy, History, LogOut, User } from 'lucide-react';
import { useAuthStore } from '../store/authStore';

const Navbar = () => {
  const { user, logout, isAuthenticated } = useAuthStore();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <nav className="bg-indigo-600 text-white shadow-lg">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Trophy className="h-8 w-8" />
            <span className="font-bold text-xl">IPL Predictor</span>
          </Link>
          
          {isAuthenticated ? (
            <div className="flex items-center space-x-6">
              <Link to="/" className="hover:text-indigo-200 transition-colors">
                Home
              </Link>
              <Link to="/predictions" className="hover:text-indigo-200 transition-colors">
                Predictions
              </Link>
              <Link to="/prediction-history" className="hover:text-indigo-200 transition-colors flex items-center space-x-1">
                <History className="h-4 w-4" />
                <span>History</span>
              </Link>
              {user?.role === 'business' && (
                <>
                  <Link to="/statistics" className="hover:text-indigo-200 transition-colors">
                    Statistics
                  </Link>
                  <Link to="/team-analysis" className="hover:text-indigo-200 transition-colors">
                    Team Analysis
                  </Link>
                </>
              )}
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-1">
                  <User className="h-4 w-4" />
                  <span>{user?.name}</span>
                </div>
                <button
                  onClick={handleLogout}
                  className="flex items-center space-x-1 hover:text-indigo-200 transition-colors"
                >
                  <LogOut className="h-4 w-4" />
                  <span>Logout</span>
                </button>
              </div>
            </div>
          ) : (
            <div className="flex space-x-4">
              <Link to="/login" className="hover:text-indigo-200 transition-colors">
                Login
              </Link>
              <Link to="/register" className="hover:text-indigo-200 transition-colors">
                Register
              </Link>
            </div>
          )}
        </div>
      </div>
    </nav>
  );
};

export default Navbar;