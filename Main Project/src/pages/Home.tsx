import React from 'react';
import { TrendingUp, Award, BarChart2 } from 'lucide-react';

const Home = () => {
  return (
    <div className="space-y-8">
      <header className="text-center">
        <h1 className="text-4xl font-bold text-gray-900 mb-4">
          IPL Cricket Match Prediction System
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Advanced machine learning and deep learning powered predictions for IPL cricket matches
        </p>
      </header>

      <div className="grid md:grid-cols-3 gap-8 mt-12">
        <FeatureCard
          icon={<TrendingUp className="h-8 w-8 text-indigo-600" />}
          title="ML Predictions"
          description="Get accurate match predictions powered by advanced machine learning algorithms"
        />
        <FeatureCard
          icon={<BarChart2 className="h-8 w-8 text-indigo-600" />}
          title="Deep Analytics"
          description="Explore comprehensive team and player statistics with interactive visualizations"
        />
        <FeatureCard
          icon={<Award className="h-8 w-8 text-indigo-600" />}
          title="Team Analysis"
          description="In-depth analysis of team performance and historical match data"
        />
      </div>

      <div className="mt-12 bg-white rounded-lg shadow-xl overflow-hidden">
        <img 
          src="https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?auto=format&fit=crop&w=1200&q=80"
          alt="Cricket Stadium"
          className="w-full h-64 object-cover"
        />
      </div>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <div className="flex items-center justify-center mb-4">
        {icon}
      </div>
      <h3 className="text-xl font-semibold text-gray-900 text-center mb-2">{title}</h3>
      <p className="text-gray-600 text-center">{description}</p>
    </div>
  );
};

export default Home;