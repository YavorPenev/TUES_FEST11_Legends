import { React, StrictMode } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Route, Routes } from 'react-router';
import Home from './home';
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
import Articles from './articles';
import ArticleDetails from './ArticleDetails';
import Calculator from './assets/SimpCalc';

import './styles/index.css';

//tuk se importwa samo home
// da ne se slusha gorniq komentar, wsi`ko se importwa tuk
// saamo tuk wsi`ko se importwa i nikude drugade

ReactDOM.createRoot(document.getElementById('root')).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home/>}/>
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
        <Route path="/articles" element={<Articles />} />
        <Route path="/article" element={<ArticleDetails />} />
        <Route path="/SimpCalc" element={<Calculator />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);