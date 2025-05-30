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
import News from './articles';
import ArticleDetails from './assets/ArticleDetails';
import Profile from './profile';
import PerArticles from './PerArticles';
import './styles/index.css';
import Sitelinks from './UsefulSourcesPage';
import LearnMore from './LearnMore';
import BudgetPlanner from './budgetplanner'
import StockPrices from './stock';

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
        <Route path="/articles" element={<News />} />
        <Route path="/article" element={<ArticleDetails />} />
        <Route path="/profile" element={<Profile />} />
        <Route path="/PerArticles" element={<PerArticles/>}/>
        <Route path="/usefulsources" element={<Sitelinks/>}/> 
        <Route path="/learnmore" element={<LearnMore/>}/>
        <Route path="/budgetplanner" element = {<BudgetPlanner />}/>
        <Route path="/stocks" element={<StockPrices />} />
      </Routes>
    </BrowserRouter>
  </StrictMode>
);