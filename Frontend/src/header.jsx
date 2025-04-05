import {React, useState, useRef} from "react";
import { Link } from "react-router";


function Header() {
    const calcmenuref = useRef(null);
    const [CalcStatus, SetCalcStatus] = useState(false);

    const CalcMenuChange = () => {
        SetCalcStatus(!CalcStatus);
        if (calcmenuref.current) {
          calcmenuref.current.style.display = CalcStatus ? 'none' : 'flex';
        }
      };

    return (
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

                <h1 className="text-white font-bold text-4xl ml-4">SmartBudget</h1>
            </div>

            <div className='flex flex-row justify-end gap-0 items-center'>
                <Link
                    to="/articles"
                    className="bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
                >
                    Articles
                </Link>
                <Link
                    to="/login"
                    className="bg-blue-100 ml-15 pl-3 pr-3 pt-2 pb-2 rounded-l-2xl border-blue-900 border-r-2 text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
                >
                    Login
                </Link>
                <Link
                    to="/signup"
                    className="bg-blue-100 ml-0 pl-3 pr-3 pt-2 pb-2 rounded-r-2xl border-l-2 border-blue-900 text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
                >
                    Sign Up
                </Link>
                <Link
                    to="/account"
                    className="bg-blue-100 ml-15 p-1 rounded-full text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200"
                ><img src='/proficon.png' className='h-13 w-13' />
                </Link>

                <button
                    className="relative ml-15 p-3 rounded-2xl bg-blue-100 hover:scale-110 transition-transform duration-200"
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
    );
}

export default Header;
