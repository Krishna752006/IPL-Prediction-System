import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuthStore } from '../store/authStore';
import { LogIn } from 'lucide-react';
import API from "../api/authApi";

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [userType, setUserType] = useState('user');
  const [showGuestModal, setShowGuestModal] = useState(false);
  const navigate = useNavigate();
  const login = useAuthStore((state) => state.login);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const response = await API.post("/login", {
        email,
        password
      });
      if (response.data.success) {
        login(
            response.data.user,
            response.data.access_token
        );
        navigate("/");
      } else { alert(response.data.message); }
  } catch (error: any) {
    if (error.response?.data?.detail) {
      alert(error.response.data.detail);
    } else if (error.response?.data?.message) {
      alert(error.response.data.message);
    } else {
      alert("Something went wrong.");
    }
  }
  };

  const handleGuestLogin = (role: "user" | "business") => {
    login(
      {
        id: role === "user" ? "guest-user" : "guest-business",
        name: role === "user" ? "Guest User" : "Business Guest",
        email: role === "user"
          ? "guest@ipl.com"
          : "business@ipl.com",
        role,
        isGuest: true,
      },
      "GUEST_SESSION"
    );

    setShowGuestModal(false);
    navigate("/");
  };

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <div className="max-w-md w-full bg-white p-8 rounded-lg shadow-lg">
        <div className="flex justify-center mb-6">
          <LogIn className="h-12 w-12 text-indigo-600" />
        </div>
        <h2 className="text-2xl font-bold text-center text-gray-900 mb-8">
          Sign in to your account
        </h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email address
            </label>
            <input
              id="email"
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-2 border"
              placeholder="Enter your email"
            />
          </div>
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-700">
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 p-2 border"
              placeholder="Enter your password"
            />
          </div>

          <button
            type="submit"
            className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
          >
            Sign in
          </button>

          <div className="mt-4">
            <button
                type="button"
                onClick={() => setShowGuestModal(true)}
                className="w-full py-2 px-4 rounded-md border border-gray-300 bg-white text-gray-700 hover:bg-gray-100 transition"
              >
                Continue as Guest
              </button>
          </div>

          <div className="text-center text-sm text-gray-600">
            Don't have an account?{' '}
            <Link to="/register" className="text-indigo-600 hover:text-indigo-500">
              Register now
            </Link>
          </div>
        </form>
      </div>

      {showGuestModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50">

          <div className="bg-white rounded-xl shadow-xl w-96 p-6">

            <h2 className="text-2xl font-bold text-center">
              Continue as Guest
            </h2>

            <p className="text-gray-500 text-center mt-2 mb-6">
              Choose how you want to explore the application.
            </p>

            <div className="space-y-3">

              <button
                type="button"
                onClick={() => handleGuestLogin("user")}
                className="w-full py-3 rounded-lg bg-indigo-600 text-white hover:bg-indigo-700"
              >
                👤 Regular User
              </button>

              <button
                type="button"
                onClick={() => handleGuestLogin("business")}
                className="w-full py-3 rounded-lg bg-green-600 text-white hover:bg-green-700"
              >
                🏢 Business User
              </button>

              <button
                type="button"
                onClick={() => setShowGuestModal(false)}
                className="w-full py-3 rounded-lg border hover:bg-gray-100"
              >
                Cancel
              </button>

            </div>

          </div>

        </div>
      )}

    </div>
  );
};

export default Login;