import React, { useState } from 'react';
import { Users, Trophy, TrendingUp } from 'lucide-react';

const TeamAnalysis = () => {
  const [selectedTeam, setSelectedTeam] = useState('Chennai Super Kings');

  const teams = {
    'Chennai Super Kings': {
      color: '#ffd700',
      titles: 5,
      winRate: 59.8,
      keyPlayers: ['MS Dhoni', 'Ravindra Jadeja', 'Ruturaj Gaikwad'],
      strengths: ['Consistent Performance', 'Strong Home Record', 'Experienced Squad'],
      weaknesses: ['Aging Squad', 'Overseas Dependency', 'Death Bowling'],
      players: [
        { name: 'MS Dhoni', price: 'Retained' },
        { name: 'Ravindra Jadeja', price: 'Retained' },
        { name: 'Ruturaj Gaikwad', price: 'Retained' },
        { name: 'Deepak Chahar', price: 'Rs 14 Crore' }, 
        { name: 'Devon Conway', price: 'Rs 6.25 Crore' },
        { name: 'Rahul Tripathi', price: 'Rs 3.4 Crore' },
        { name: 'Rachin Ravindra', price: 'Rs 4 Crore' },
        { name: 'Ravichandran Ashwin', price: 'Rs 9.75 Crore' },
        { name: 'Khaleel Ahmed', price: 'Rs 4.80 Crore' },
        { name: 'Noor Ahmed', price: 'Rs 10 Crore' },
        { name: 'Vijay Shankar', price: 'Rs 1.2 Crore' },
        { name: 'Sam Curran', price: 'Rs 2.4 Crore' },
        { name: 'Shaik Rasheed', price: 'Rs 30 Lakh' },
        { name: 'Anshul Kamboj', price: 'Rs 3.4 Crore' },
        { name: 'Mukesh Choudhary', price: 'Rs 30 Lakh' },
        { name: 'Deepak Hooda', price: 'Rs 1.7 Crore' },
        { name: 'Gurjapneet Singh', price: 'Rs 2.2 Crore' },
        { name: 'Nathan Ellis', price: 'Rs 2 Crore' },
        { name: 'Kamlesh Nagarkoti', price: 'Rs 30 Lakh' },
        { name: 'Ramakrishna Ghosh', price: 'Rs 30 Lakh' },
        { name: 'Shreyas Gopal', price: 'Rs 30 Lakh' },
        { name: 'Vansh Bedi', price: 'Rs 55 Lakh' },
        { name: 'Andre Siddarth', price: 'Rs 30 Lakh' },
        { name: 'Matheesha Pathirana', price: 'Retained' },
        { name: 'Shivam Dube', price: 'Retained' }
      ]
    },
    'Mumbai Indians': {
      color: '#004ba0',
      titles: 5,
      winRate: 57.2,
      keyPlayers: ['Rohit Sharma', 'Jasprit Bumrah', 'Suryakumar Yadav','Trent Boult'],
      strengths: ['Death Bowling', 'Power Hitting', 'Strong Core Team'],
      weaknesses: ['Middle Overs Batting', 'Spin Bowling Options', 'Inconsistent Openers'],
      players: [ { name: 'Rohit Sharma', price: 'Retained' }, { name: 'Jasprit Bumrah', price: 'Retained' }, { name: 'Suryakumar Yadav', price: 'Retained' }, { name: 'Trent Boult', price: 'Rs 12.5 Crore' }, { name: 'Naman Dhir', price: 'Rs 5.25 Crore' }, { name: 'Robin Minz', price: 'Rs 65 lakh' }, { name: 'Karn Sharma', price: 'Rs 50 lakh' }, { name: 'Ryan Rickelton', price: 'Rs 1 crore' }, { name: 'Deepak Chahar', price: 'Rs 9.25 crore' }, { name: 'Allah Ghazanfar', price: 'Rs 4.8 crore' }, { name: 'Will Jacks', price: 'Rs 5.25 crore' }, { name: 'Ashwani Kumar', price: 'Rs 30 lakh' }, { name: 'Mitchell Santner', price: 'Rs 2 crore' }, { name: 'Reece Topley', price: 'Rs 75 lakh' }, { name: 'Shrijith Krishnan', price: 'Rs 30 lakh' }, { name: 'Raj Angad Bawa', price: 'Rs 30 lakh' }, { name: 'Satyanarayana Raju', price: 'Rs 30 lakh' }, { name: 'Bevon Jacobs', price: 'Rs 30 lakh' }, { name: 'Arjun Tendulkar', price: 'Rs 30 lakh' }, { name: 'Lizaad Williams', price: 'Rs 75 lakh' }, { name: 'Vignesh Puthur', price: 'Rs 30 lakh' }, { name: 'Tilak Varma', price: 'Retained' }, { name: 'Hardik Pandya', price: 'Retained' }]
    },
    'Royal Challengers Bangalore': {
      color: '#aa2020',
      titles: 0,
      winRate: 48.4,
      keyPlayers: ['Virat Kohli', 'Phill Salt', 'Bhuvneshwar Kumar'],
      strengths: ['Strong Batting Lineup', 'Power Hitters', 'Star Players'],
      weaknesses: ['Inconsistent Bowling', 'Death Over Issues', 'Dependence on Key Players'],
      players: [
        { name: 'Liam Livingstone', price: 'Rs 8.75 crore' },
        { name: 'Phil Salt', price: 'Rs 11.50 crore' },
        { name: 'Jitesh Sharma', price: 'Rs 11 crore' },
        { name: 'Josh Hazlewood', price: 'Rs 12.5 crore' },
        { name: 'Rasikh Dar', price: 'Rs 6 crore' },
        { name: 'Suyash Sharma', price: 'Rs 2.6 crore' },
        { name: 'Krunal Pandya', price: 'Rs 5.75 crore' },
        { name: 'Bhuvneshwar Kumar', price: 'Rs 10.75 crore' },
        { name: 'Swapnil Singh', price: 'Rs 50 lakh' },
        { name: 'Tim David', price: 'Rs 3 crore' },
        { name: 'Romario Shepherd', price: 'Rs 1.5 crore' },
        { name: 'Nuwan Thushara', price: 'Rs 1.6 crore' },
        { name: 'Manoj Bhandage', price: 'Rs 30 lakh' },
        { name: 'Jacob Bethell', price: 'Rs 2.6 crore' },
        { name: 'Devdutt Padikkal', price: 'Rs 2 crore' },
        { name: 'Swastik Chikara', price: 'Rs 30 lakh' },
        { name: 'Lungi Ngidi', price: 'Rs 1 crore' },
        { name: 'Abhinandan Singh', price: 'Rs 30 lakh' },
        { name: 'Mohit Rathee', price: 'Rs 30 lakh' },
        { name: 'Virat Kohli', price: 'Rs 21 crore' },
        { name: 'Rajat Patidar', price: 'Rs 11 crore' },
        { name: 'Yash Dayal', price: 'Rs 5 crore' }
      ]
      
    },
    'Delhi Capitals': {
      color: '#0047AB',
      titles: 0,
      winRate: 44.5,
      keyPlayers: ['Starc', 'KL Rahul', 'Axar Patel'],
      strengths: ['Strong Bowling Attack', 'Explosive Openers', 'Young Squad'],
      weaknesses: ['Inconsistent Middle Order', 'Dependence on Key Players'],
      players: [
        { name: 'Mitchell Starc', price: 'Rs 11.75 cr' },
        { name: 'KL Rahul', price: 'Rs 14 cr' },
        { name: 'Harry Brook', price: 'Rs 6.25 cr' },
        { name: 'Jake Fraser-McGurk', price: 'Rs 9 cr' },
        { name: 'T Natarajan', price: 'Rs 10.75 cr' },
        { name: 'Karun Nair', price: 'Rs 50 lakh' },
        { name: 'Sameer Rizvi', price: 'Rs 95 lakh' },
        { name: 'Ashutosh Sharma', price: 'Rs 3.8 crore' },
        { name: 'Mohit Sharma', price: 'Rs 2.2 crore' },
        { name: 'Faf Du Plessis', price: 'Rs 2 crore' },
        { name: 'Mukesh Kumar', price: 'Rs 8 crore' },
        { name: 'Darshan Nalkande', price: 'Rs 30 lakh' },
        { name: 'Dushmantha Chameera', price: 'Rs 75 lakh' },
        { name: 'Donovan Ferreira', price: 'Rs 75 lakh' },
        { name: 'Ajay Mandal', price: 'Rs 30 lakh' },
        { name: 'Manvanth Kumar L', price: 'Rs 30 lakh' },
        { name: 'Tripurana Vijay', price: 'Rs 30 lakh' },
        { name: 'Madhav Tiwari', price: 'Rs 40 lakh' },
        { name: 'Axar Patel', price: 'Rs 16.5 cr' },
        { name: 'Kuldeep Yadav', price: 'Rs 13.25 cr' },
        { name: 'Tristan Stubbs', price: 'Rs 10 cr' },
        { name: 'Abishek Porel', price: 'Rs 4 cr (uncapped)' },
        { name: 'Rishabh Pant', price: 'Retained' },
        { name: 'Prithvi Shaw', price: 'Rs 7.5 Crore' },
        { name: 'Anrich Nortje', price: 'Rs 6.5 Crore' }
      ]
      
    },
    'Kolkata Knight Riders': {
      color: '#3A225D',
      titles: 2,
      winRate: 51.3,
      keyPlayers: ['Rinku Singh', 'Andre Russell', 'Sunil Narine'],
      strengths: ['All-Rounders', 'Spin Attack', 'Explosive Finishers'],
      weaknesses: ['Inconsistent Openers', 'Death Bowling'],
      players: [
        { name: 'Shreyas Iyer', price: 'Retained' },
        { name: 'Andre Russell', price: 'Rs 12 Crore' },
        { name: 'Sunil Narine', price: 'Rs 10 Crore' },
        { name: 'Rinku Singh', price: 'Rs 5 crore' },
        { name: 'Varun Chakravarthy', price: 'Rs 8 crore' },
        { name: 'Harshit Rana', price: 'Rs 2 crore' },
        { name: 'Ramandeep Singh', price: 'Rs 3 crore' },
        { name: 'Venkatesh Iyer', price: 'Rs 6.5 crore' },
        { name: 'Quinton de Kock', price: 'Rs 9 crore' },
        { name: 'Rahmanullah Gurbaz', price: 'Rs 4 crore' },
        { name: 'Anrich Nortje', price: 'Rs 6 crore' },
        { name: 'Angkrish Raghuvanshi', price: 'Rs 2.5 crore' },
        { name: 'Vaibhav Arora', price: 'Rs 3 crore' },
        { name: 'Mayank Markande', price: 'Rs 4 crore' },
        { name: 'Rovman Powell', price: 'Rs 7 crore' },
        { name: 'Manish Pandey', price: 'Rs 8 crore' },
        { name: 'Spencer Johnson', price: 'Rs 2 crore' },
        { name: 'Luvnith Sisodia', price: 'Rs 1 crore' },
        { name: 'Ajinkya Rahane', price: 'Rs 10 crore' },
        { name: 'Anukul Roy', price: 'Rs 3 crore' },
        { name: 'Moeen Ali', price: 'Rs 6 crore' },
        { name: 'Umran Malik', price: 'Rs 5 crore' }
      ]      
    },
    'Punjab Kings': {
      color: '#D71920',
      titles: 0,
      winRate: 47.8,
      keyPlayers: ['Shreyas Iyer', 'Chahal', 'Arshadeep Shing'],
      strengths: ['Strong Batting Lineup', 'Power Hitters'],
      weaknesses: ['Inconsistent Bowling', 'Lack of Experience'],
      players: [
        { name: 'Arshdeep Singh', price: 'Rs 18 Crore (RTM)' },
        { name: 'Shreyas Iyer', price: 'Rs 26.75 Crore' },
        { name: 'Yuzvendra Chahal', price: 'Rs 18 Crore' },
        { name: 'Marcus Stoinis', price: 'Rs 11 Crore' },
        { name: 'Glenn Maxwell', price: 'Rs 4.2 Crore' },
        { name: 'Nehal Wadhera', price: 'Rs 4.2 Crore' },
        { name: 'Harpreet Brar', price: 'Rs 1.5 Crore' },
        { name: 'Vishnu Vinod', price: 'Rs 95 Lakh' },
        { name: 'Vijaykumar Vyshak', price: 'Rs 1.8 Crore' },
        { name: 'Yash Thakur', price: 'Rs 1.8 Crore' },
        { name: 'Marco Jansen', price: 'Rs 7 Crore' },
        { name: 'Josh Inglis', price: 'Rs 2.6 Crore' },
        { name: 'Lockie Ferguson', price: 'Rs 2 Crore' },
        { name: 'Azmatullah Omarzai', price: 'Rs 2.4 Crore' },
        { name: 'Harnoor Pannu', price: 'Rs 30 Lakh' },
        { name: 'Kuldeep Sen', price: 'Rs 80 Lakh' },
        { name: 'Priyansh Arya', price: 'Rs 3.8 Crore' },
        { name: 'Aaron Hardie', price: 'Rs 1.25 Crore' },
        { name: 'Musheer Khan', price: 'Rs 30 Lakh' },
        { name: 'Suryansh Shedge', price: 'Rs 30 Lakh' },
        { name: 'Xavier Bartlett', price: 'Rs 80 Lakh' },
        { name: 'Pyla Avinash', price: 'Rs 30 Lakh' },
        { name: 'Praveen Dubey', price: 'Rs 30 Lakh' },
        { name: 'Prabhsimran Singh', price: 'Rs 2 Crore' },
        { name: 'Shashank Singh', price: 'Rs 1.5 Crore' }
      ]
      
    },
    'Rajasthan Royals': {
      color: '#FF8C00',
      titles: 1,
      winRate: 52.5,
      keyPlayers: ['Sanju Samson', 'Jos Buttler', 'Yashasvi Jaiswal'],
      strengths: ['Strong Batting', 'Best All-Rounders', 'Top Pace Attack'],
      weaknesses: ['Inconsistent Death Over Bowling', 'Dependence on Jos Buttler'],
      players: [
        { name: 'Jofra Archer', price: 'Rs 12.5 Crore' },
        { name: 'Maheesh Theekshana', price: 'Rs 4.4 Crore' },
        { name: 'Wanindu Hasaranga', price: 'Rs 5.25 Crore' },
        { name: 'Akash Madhwal', price: 'Rs 1.2 Crore' },
        { name: 'Kumar Kartikeya', price: 'Rs 30 Lakh' },
        { name: 'Nitish Rana', price: 'Rs 4.4 Crore' },
        { name: 'Tushar Deshpande', price: 'Rs 6.5 Crore' },
        { name: 'Shubham Dubey', price: 'Rs 80 Lakh' },
        { name: 'Yudhvir Singh', price: 'Rs 35 Lakh' },
        { name: 'Fazalhaq Farooqi', price: 'Rs 2 Crore' },
        { name: 'Vaibhav Suryavanshi', price: 'Rs 1.1 Crore' },
        { name: 'Kwena Maphaka', price: 'Rs 1.5 Crore' },
        { name: 'Kunal Rathore', price: 'Rs 30 Lakh' },
        { name: 'Ashok Sharma', price: 'Rs 30 Lakh' },
        { name: 'Sanju Samson', price: 'Rs 14 Crore' },
        { name: 'Yashasvi Jaiswal', price: 'Rs 5.5 Crore' },
        { name: 'Riyan Parag', price: 'Rs 3.8 Crore' },
        { name: 'Sandeep Sharma', price: 'Rs 1.2 Crore' },
        { name: 'Shimron Hetmyer', price: 'Rs 8.5 Crore' },
        { name: 'Dhruv Jurel', price: 'Rs 1 Crore' }
      ]
      
    },
    'Sunrisers Hyderabad': {
      color: '#F66000',
      titles: 1,
      winRate: 54.3,
      keyPlayers: ['Travis Head', 'Abhishek', 'Pat Cummins','Nithish Kumar Reddy'],
      strengths: ['Strong Bowling Attack', 'Top-Class Fielding'],
      weaknesses: ['Top-Order Dependency', 'Lack of All-Rounders'],
      players: [
        { name: 'Mohammed Shami', price: 'Rs 10 Crore' },
        { name: 'Harshal Patel', price: 'Rs 8 Crore' },
        { name: 'Ishan Kishan', price: 'Rs 11.25 Crore' },
        { name: 'Rahul Chahar', price: 'Rs 3.2 Crore' },
        { name: 'Adam Zampa', price: 'Rs 2.4 Crore' },
        { name: 'Atharva Taide', price: 'Rs 30 Lakh' },
        { name: 'Abhinav Manohar', price: 'Rs 3.2 Crore' },
        { name: 'Simarjeet Singh', price: 'Rs 1.5 Crore' },
        { name: 'Zeeshan Ansari', price: 'Rs 40 Lakh' },
        { name: 'Jaydev Unadkat', price: 'Rs 1 Crore' },
        { name: 'Brydon Carse', price: 'Rs 1 Crore' },
        { name: 'Kamindu Mendis', price: 'Rs 75 Lakh' },
        { name: 'Aniket Verma', price: 'Rs 30 Lakh' },
        { name: 'Eshan Malinga', price: 'Rs 1.2 Crore' },
        { name: 'Sachin Baby', price: 'Rs 30 Lakh' },
        { name: 'Heinrich Klaasen', price: 'Rs 9.5 Crore' },
        { name: 'Pat Cummins', price: 'Rs 15 Crore' },
        { name: 'Abhishek Sharma', price: 'Rs 4 Crore' },
        { name: 'Travis Head', price: 'Rs 7.5 Crore' },
        { name: 'Nitish Kumar', price: 'Rs 3.8 Crore' }
      ]
      
    },
    'Lucknow Super Gaints': {
      color: '#F46100',
      titles: 0,
      winRate: 54.3,
      keyPlayers: ['Pant', 'Pooran', 'Mayank'],
      strengths: ['Strong Bowling Attack', 'Top-Class Fielding'],
      weaknesses: ['Top-Order Dependency', 'Lack of All-Rounders'],
      players: [
        { name: 'Rishabh Pant', price: 'Rs 27 Crore' },
        { name: 'Aiden Markram', price: 'Rs 2 Crore' },
        { name: 'David Miller', price: 'Rs 7.5 Crore' },
        { name: 'Mitchell Marsh', price: 'Rs 3.4 Crore' },
        { name: 'Avesh Khan', price: 'Rs 9.75 Crore' },
        { name: 'Abdul Samad', price: 'Rs 4.2 Crore' },
        { name: 'Aryan Juyal', price: 'Rs 30 Lakh' },
        { name: 'Akash Deep', price: 'Rs 8 Crore' },
        { name: 'Himmat Singh', price: 'Rs 30 Lakh' },
        { name: 'M Siddharth', price: 'Rs 75 Lakh' },
        { name: 'Digvesh Singh', price: 'Rs 35 Lakh' },
        { name: 'Shahbaz Ahmed', price: 'Rs 2.4 Crore' },
        { name: 'Akash Singh', price: 'Rs 30 Lakh' },
        { name: 'Shamar Joseph', price: 'Rs 75 Lakh' },
        { name: 'Prince Yadav', price: 'Rs 30 Lakh' },
        { name: 'Yuvraj Chaudhary', price: 'Rs 30 Lakh' },
        { name: 'Rajvardhan Hangargekar', price: 'Rs 30 Lakh' },
        { name: 'Arshin Kulkarni', price: 'Rs 30 Lakh' },
        { name: 'Matthew Breetzke', price: 'Rs 75 Lakh' },
        { name: 'Nicholas Pooran', price: 'TBD' },
        { name: 'Mayank Yadav', price: 'TBD' },
        { name: 'Ravi Bishnoi', price: 'TBD' },
        { name: 'Mohsin Khan', price: 'TBD' },
        { name: 'Ayush Badoni', price: 'TBD' }
      ]
      
    },
    'Gujarat Titains': {
      color: '#004ba0',
      titles: 1,
      winRate: 57.2,
      keyPlayers: ['Jos Buttler', 'Subman Gill', 'Mohammed Siraj','Kagiso Rabada'],
      strengths: ['Death Bowling', 'Power Hitting', 'Strong Core Team'],
      weaknesses: ['Middle Overs Batting', 'Spin Bowling Options', 'Inconsistent Openers'],
      players: [
        { name: 'Kagiso Rabada', price: 'Rs 10.75 Crore' },
        { name: 'Jos Buttler', price: 'Rs 15.75 Crore' },
        { name: 'Mohammed Siraj', price: 'Rs 12.25 Crore' },
        { name: 'Prasidh Krishna', price: 'Rs 9.50 Crore' },
        { name: 'Nishant Sindhu', price: 'Rs 30 Lakh' },
        { name: 'Mahipal Lomror', price: 'Rs 1.7 Crore' },
        { name: 'Kumar Kushagra', price: 'Rs 65 Lakh' },
        { name: 'Anuj Rawat', price: 'Rs 30 Lakh' },
        { name: 'Manav Suthar', price: 'Rs 30 Lakh' },
        { name: 'Washington Sundar', price: 'Rs 3.20 Crore' },
        { name: 'Gerald Coetzee', price: 'Rs 2.40 Crore' },
        { name: 'Arshad Khan', price: 'Rs 1.3 Crore' },
        { name: 'Gurnoor Brar', price: 'Rs 1.3 Crore' },
        { name: 'Sherfane Rutherford', price: 'Rs 2.6 Crore' },
        { name: 'Sai Kishore', price: 'Rs 2 Crore' },
        { name: 'Ishant Sharma', price: 'Rs 75 Lakh' },
        { name: 'Jayant Yadav', price: 'Rs 75 Lakh' },
        { name: 'Glenn Phillips', price: 'Rs 2 Crore' },
        { name: 'Karim Janat', price: 'Rs 75 Lakh' },
        { name: 'Kulwant Khejroliya', price: 'Rs 30 Lakh' },
        { name: 'Rashid Khan', price: 'TBD' },
        { name: 'Shubman Gill', price: 'TBD' },
        { name: 'B Sai Sudharsan', price: 'TBD' },
        { name: 'Rahul Tewatia', price: 'TBD' },
        { name: 'Shahrukh Khan', price: 'TBD' }
      ]      
    }

  };

  const selectedTeamData = teams[selectedTeam];
  const halfIndex = Math.ceil(selectedTeamData.players.length / 2);
  const firstHalf = selectedTeamData.players.slice(0, halfIndex);
  const secondHalf = selectedTeamData.players.slice(halfIndex);   

  return (
    <div className="space-y-8">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Team Analysis</h2>
        <select
          value={selectedTeam}
          onChange={(e) => setSelectedTeam(e.target.value)}
          className="p-2 border rounded-md"
        >
          {Object.keys(teams).map(team => (
            <option key={team} value={team}>{team}</option>
          ))}
        </select>
      </div>

      <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
        <StatCard
          icon={<Trophy className="h-6 w-6" />}
          title="IPL Titles"
          value={selectedTeamData.titles}
          color={selectedTeamData.color}
        />
        <StatCard
          icon={<TrendingUp className="h-6 w-6" />}
          title="Win Rate"
          value={`${selectedTeamData.winRate}%`}
          color={selectedTeamData.color}
        />
        <StatCard
          icon={<Users className="h-6 w-6" />}
          title="Squad Size"
          value={selectedTeamData.players.length}
          color={selectedTeamData.color}
        />
      </div>

      <div className="grid md:grid-cols-2 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">Key Players</h3>
          <div className="space-y-4">
            {selectedTeamData.keyPlayers.map((player, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: selectedTeamData.color }} />
                <span>{player}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white p-6 rounded-lg shadow-lg">
          <h3 className="text-xl font-semibold mb-4">Team Analysis</h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Strengths</h4>
              <ul className="list-disc list-inside space-y-1">
                {selectedTeamData.strengths.map((strength, index) => (
                  <li key={index}>{strength}</li>
                ))}
              </ul>
            </div>
            <div>
              <h4 className="font-medium text-gray-700 mb-2">Areas for Improvement</h4>
              <ul className="list-disc list-inside space-y-1">
                {selectedTeamData.weaknesses.map((weakness, index) => (
                  <li key={index}>{weakness}</li>
                ))}
              </ul>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-white p-6 rounded-lg shadow-lg">
        <h3 className="text-xl font-semibold mb-4">Squad Players</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ul className="list-disc list-inside space-y-2">
            {firstHalf.map((player, index) => (
              <li key={index}>
                {player.name} - <span className="text-gray-600">{player.price}</span>
              </li>
            ))}
          </ul>
          <ul className="list-disc list-inside space-y-2">
            {secondHalf.map((player, index) => (
              <li key={index}>
                {player.name} - <span className="text-gray-600">{player.price}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, title, value, color }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-lg">
      <div className="flex items-center justify-between mb-4">
        <div style={{ color }}>{icon}</div>
        <span className="text-2xl font-bold" style={{ color }}>{value}</span>
      </div>
      <h3 className="text-gray-600">{title}</h3>
    </div>
  );
};

export default TeamAnalysis;