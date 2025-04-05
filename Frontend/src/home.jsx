import { useState, useRef } from 'react';
import { Routes, Route, useNavigate, Link, useLocation } from 'react-router-dom';
import './styles/index.css';
import About from './about';
import FAQ from './FAQ';
import Law from './law';
import AdviceAI from './adviceAI';
import InvestAI from './InvestAI';
import InvestCalc from './InvCalc';
import LoanCalc from './loancalc';
import CurrencyCalc from './CurenCalc';
import Login from './login';
import SignUP from './signup';

function Home() {
  const navigate = useNavigate();
  const location = useLocation();
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

  //pt-24 - na wsqka stranica za da raboti hedyra
  const hideHeader = location.pathname === '/login' || location.pathname === '/signup';

  return (
    <>

      {!hideHeader && (
        <header className="pl-5 pr-5 fixed top-0 left-0 w-full bg-blue-800 border-blue-900 flex justify-between p-4 border-b-4 bg-gradient-to-b from-gray-800 to-transparent items-center z-50">
          <div className='h-[100%] flex flex-row justify-start items-center'>
            <Link to="/" className="bg-blue-100 pr-3 pl-3 pt-1.5 pb-1.5 rounded-xl hover:scale-110 transition-transform duration-200">
              <img
                title="SmartBudget"
                alt="logo"
                src="/logo2.png"
                className="max-h-12 aspect-auto"
              />
            </Link>

            <h1 className="text-white font-bold text-3xl ml-4">SmartBudget</h1>
          </div>

          <div className='flex flex-row justify-end gap-4 items-center'>
            <Link
              to="/articles"
              className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
            >
              Articles
            </Link>
            <Link
              to="/login"
              className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
            >
              Login
            </Link>
            <Link
              to="/signup"
              className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
            >
              Sign Up
            </Link>
            <Link
              to="/account"
              className="bg-blue-100 p-1 rounded-full text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
            ><img src='/proficon.png' className='h-13 w-13' />
            </Link>

            <button
              className="relative p-3 rounded-2xl bg-blue-100 hover:scale-110 transition-transform duration-200"
              onClick={CalcMenuChange}
            >
              <img src="/calclogo.png" alt="Calculator" className="h-9" />
            </button>
            <div
              ref={calcmenuref}
              className="absolute top-full mt-0 flex-col rounded-bl-xl bg-blue-800 border-b-4 border-r-0 border-l-4 border-blue-950 right-0 p-2 gap-2 w-25 text-center text-blue-100 shadow-lg"
              style={{ display: 'none' }}
            >
              <Link
                className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
                to="/calcloan"
                title="Advanced Loan Calculator"
              >
                Loan
              </Link>
              <Link
                className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
                to="/calcstocks"
                title="Stocks Profit Calculator"
              >
                Stocks
              </Link>
              <Link
                className="bg-blue-100 text-blue-950 rounded-lg p-2 hover:scale-110 transition-transform duration-200"
                to="/CurrencyCalculator"
                title="Currency Exchange Calculator"
              >
                Currency
              </Link>
            </div>
          </div>
        </header>
      )}

    <Routes>
  <Route
    path="/"
    element={
     <div className="pt-20">
        <div
          ref={aimenuref}
          className="flex-col-reverse rounded-t-full fixed right-5 bottom-17 pb-15 pt-5 gap-4 text-blue-100 w-25 bg-blue-800 border-4 border-blue-950"
          style={{ display: 'none' }}
        >
          <Link
            className="w-[80%] ml-[10%] mr-[10%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform duration-200"
            to="/invest"
          >
            Investment<br />
            advisor
          </Link>
          <Link
            className="w-[80%] ml-[10%] mr-[10%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform duration-200"
            to="/advice"
          >
            Stocks<br />
            Advisor
          </Link>
          <Link
            className="w-[80%] ml-[10%] mr-[10%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform duration-200"
            to="/budgetplaner"
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
                className="-mt-20 w-full border-b-8 border-b-blue-900"
                src="/mainpic-above.png"
                alt="Main Visual"
              />

              <div className="flex justify-center items-center pb-5 pt-5 w-full flex-col bg-gradient-to-b from-blue-100 to-gray-400 border-b-8 border-blue-900">
                <img className="w-[30%]" src="/SmartBudget.png" alt="SmartBudget Logo" />
                <Link
                  to="/faq"
                  className="absolute top-[45%] right-4 bg-blue-900 pt-2 pb-2 pr-5 pl-5 rounded-2xl text-2xl font-extrabold text-blue-100 hover:scale-110 transition-transform duration-200"
                >
                  ?
                </Link>

                <h1 className="text-4xl font-semibold text-blue-900">
                  Your Ultimate Financial Knowledge Hub
                </h1>
                <Link
                  to="/learnmore"
                  className="bg-blue-800 mt-10 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-2xl text-blue-100 font-bold hover:scale-110 transition-transform duration-200"
                >
                  Learn More
                </Link>
              </div>

              <img className="w-full" src="/mainpic-bottom.png" alt="Footer Visual" />
            </div>

          }
        />

        <Route path="/about" element={<About />} />
        <Route path="/faq" element={<FAQ />} />
        <Route path="/law" element={<Law />} />
        <Route path="/advice" element={<AdviceAI />} />
        <Route path="/invest" element={<InvestAI />} />
        <Route path="/investcalc" element={<InvestCalc />} />
        <Route path="/calcloan" element={<LoanCalc />} />
        <Route path="/CurrencyCalculator" element={<CurrencyCalc />} />
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<SignUP />} />
      </Routes>



      {!hideHeader && (
        <footer className="flex bg-blue-800 bg-gradient-to-b from-transparent to-gray-800 text-blue-100 p-5 justify-evenly flex-wrap items-center">
          <p>©legends Development Team</p>
          <Link className="hover:underline" to="/learnmore">Learn More Page</Link>
          <Link className="hover:underline" to="/articles">Articles Page</Link>
          <Link className="align-baseline hover:underline" to="/faq">FAQ Page</Link>
          <div className="flex-col flex-nowrap justify-evenly gap-2 items-center justify-items-center">
            <a><img className="aspect-auto h-15" src="/youtube.png" alt="YouTube" /></a>
            <p>+39 06 6988 4857</p>
            <p>yavorpen@gmail.com</p>
          </div>
          <Link className="hover:underline" to="/stockai">To Stocks AI Assistant</Link>
          <Link to="/about" className="hover:underline">About</Link>
        </footer>
      )}
    </>
  );
}

export default Home;