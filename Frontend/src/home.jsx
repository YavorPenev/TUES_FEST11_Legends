import { useState, useRef } from 'react';
import { Routes, Route, useNavigate, Link } from 'react-router-dom';
import About from './about';
import FAQ from './FAQ';
import Law from './law';
import './styles/index.css';

function Home() {
  const navigate = useNavigate();
  const aimenuref = useRef(null);
  const calcmenuref = useRef(null);
  const [AiStatus, SetAiStatus] = useState(false);
  const [CalcStatus, SetCalcStatus] = useState(false);

  const AiMenuChange = () => {
    SetAiStatus(!AiStatus);
    if (aimenuref.current) {
      aimenuref.current.style.display = AiStatus ? 'none' : 'flex';
    }
  };

  const CalcMenuChange = () => {
    SetCalcStatus(!CalcStatus);
    if (calcmenuref.current) {
      calcmenuref.current.style.display = CalcStatus ? 'none' : 'flex';
    }
  };

  return (
    <>
      {/* тук Хедърът се показва на всички страници */}
      <header className="pl-5 pr-5 fixed top-0 left-0 w-full bg-blue-800 border-blue-900 flex justify-between p-4 border-b-4 bg-gradient-to-b from-gray-800 to-transparent items-center z-50">
        <Link to="/" className="bg-blue-100 pr-[1%] pl-[1%] pt-[0.4%] pb-[0.4%] rounded-xl hover:scale-110 transition-transform duration-200">
          <img
            title="SmartBudget"
            alt="logo"
            src="/logo2.png"
            className="max-h-12 aspect-auto"
          />
        </Link>

        <h1 className="text-white font-bold text-3xl ml-4">SmartBudget</h1>
        <button
          onClick={() => navigate('/articles')}
          className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
        >
          Articles
        </button>
        <button
          onClick={() => navigate('/login')}
          className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
        >
          Login
        </button>
        <button
          onClick={() => navigate('/signup')}
          className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
        >
          Sign Up
        </button>
        <button
          onClick={() => navigate('/account')}
          className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
        >
          My Account
        </button>

        <button
          className="relative p-3 rounded-2xl bg-blue-100 hover:scale-110 transition-transform duration-200"
          onClick={CalcMenuChange}
        >
          <img src="/calclogo.png" alt="Calculator" className="h-7" />
        </button>

        <div
          ref={calcmenuref}
          className="absolute top-full mt-0 flex-col rounded-b-xl bg-blue-800 border-b-4 border-r-4 border-l-4 border-blue-950 right-[9.5%] p-2 gap-2 w-25 text-center text-blue-100 shadow-lg"
          style={{ display: 'none' }}
        >
          <Link
            className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
            to="/calcloan"
          >
            Loan Calc.
          </Link>
          <Link
            className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
            to="/calcstocks"
          >
            Stocks Calc.
          </Link>
          <Link
            className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
            to="/calcexch"
          >
            Currency Exch.
          </Link>
        </div>

        <button
          onClick={() => navigate('/faq')}
          className="bg-blue-100 pt-2 pb-2 pr-5 pl-5 rounded-2xl text-2xl font-extrabold text-blue-950 hover:scale-110 transition-transform duration-200"
        >
          ?
        </button>
      </header>

 
      <div className="pt-20">
        <Routes>
    {/*тук е съдържанието на главната страница*/}
          <Route
            path="/"
            element={
              <>
     
                <div
                  ref={aimenuref}
                  className="flex-col-reverse rounded-t-full fixed right-5 bottom-17 pb-15 pt-5 gap-4 text-blue-100 w-25 bg-blue-800 border-4 border-blue-950"
                  style={{ display: 'none' }}
                >
                  <Link
                    className="w-[80%] ml-[10%] mr-[10%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform duration-200"
                    to="/stockai"
                  >
                    Stocks<br />
                    Helper
                  </Link>
                  <Link
                    className="w-[80%] ml-[10%] mr-[10%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform duration-200"
                    to="/budgetai"
                  >
                    Budget<br />
                    Planner
                  </Link>
                </div>

                <button
                  className="w-25 h-25 text-3xl pb-23 font-extrabold rounded-full border-blue-950 border-4 bg-blue-800 text-blue-100 fixed bottom-5 right-5 cursor-pointer hover:scale-110 transition-transform duration-200"
                  onClick={AiMenuChange}
                >
                  <sub>^</sub>
                  <br />
                  <b>AI</b>
                </button>

     
                <img
                  className="mt-2 w-full border-b-8 border-b-blue-900"
                  src="/mainpic-above.png"
                />

                <div className="flex justify-center items-center pb-5 pt-5 w-full flex-col bg-gradient-to-b from-blue-100 to-gray-400 border-b-8 border-blue-900">
                  <img className="w-[30%]" src="/SmartBudget.png" />
                  <h1 className="text-4xl font-semibold text-blue-900">
                    Your Ultimate Financial Knowledge Hub
                  </h1>
                  <button
                    onClick={() => navigate('/learnmore')}
                    className="bg-blue-800 mt-10 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-2xl text-blue-100 font-bold hover:scale-110 transition-transform duration-200"
                  >
                    Learn More
                  </button>
                </div>

                <img className="w-full" src="/mainpic-bottom.png" />

                <footer className="flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-between">
                  <p>©legends Development Team</p>
                </footer>
              </>
            }
          />
 
          <Route path="/about" element={<About />} />
          <Route path="/learnmore" element={<div>Learn More Page</div>} />
          <Route path="/articles" element={<div>Articles Page</div>} />
          <Route path="/faq" element={<FAQ />} />
          <Route path="/law" element={<Law />} />
        </Routes>
      </div>
    </>
  );
}

export default Home;