import { useState, useRef } from 'react';
import { Routes, Route, useNavigate, Link, useLocation } from 'react-router-dom';
import './styles/index.css';
import Header from './assets/header';
import Footer from './assets/footer';

function Home() {
  const navigate = useNavigate();
  const location = useLocation();
  const aimenuref = useRef(null);

  const [AiStatus, SetAiStatus] = useState(false);


  const AiMenuChange = () => {
    SetAiStatus(!AiStatus);
    if (aimenuref.current) {
      aimenuref.current.style.display = AiStatus ? 'none' : 'flex';
    }
  };

  //pt-24 - na wsqka stranica za da raboti hedyra
  const hideHeader = location.pathname === '/login' || location.pathname === '/signup';

  const [current, setCurrent] = useState(0);

  const slides = [
    {
      img: "car1.png",
      caption:
        "A multinational conglomerate led by Warren Buffett, known for its diverse investments in industries like insurance, energy, and consumer goods.",
    },
    {
      img: "car2.png",
      caption:
        "A leading global financial services firm offering investment banking, wealth management, and trading services to individuals, corporations, and governments.",
    },
    {
      img: "car3.png",
      caption:
        "The world's largest asset manager, specializing in investment management, risk analysis, and financial advisory services, with a focus on ETFs and sustainable investing.",
    },
  ];

  const companies = [
    {
      name: "Berkshire Hathaway",
      desc: "A multinational conglomerate led by Warren Buffett, known for its diverse investments in industries like insurance, energy, and consumer goods.",
      link: "https://www.berkshirehathaway.com/",
    },
    {
      name: "BlackRock",
      desc: "The world's largest asset management firm, with over $9 trillion in assets.",
      link: "https://www.blackrock.com/",
    },
    {
      name: "Vanguard Group",
      desc: "Famous for its low-cost funds and long-term investment strategies.",
      link: "https://www.vanguard.com/",
    },
    {
      name: "Fidelity Investments",
      desc: "A global leader in asset management and financial services.",
      link: "https://www.fidelity.com/",
    },
    {
      name: "Goldman Sachs",
      desc: "A global investment bank with significant influence on the world economy.",
      link: "https://www.goldmansachs.com/",
    },
    {
      name: "Morgan Stanley",
      desc: "One of the leading investment companies with a long history.",
      link: "https://www.morganstanley.com/",
    },
  ];

  const changeSlide = (direction) => {
    setCurrent((prev) => (prev + direction + slides.length) % slides.length);
  };

  return (
    <>
      <Header />

      <div className="pt-20">
        <div
          ref={aimenuref}
          className="z-40 flex-col-reverse rounded-t-full fixed right-5 bottom-17 pb-15 pt-10 gap-4 text-blue-100 w-25 bg-blue-800 border-4 border-blue-950"
          style={{ display: 'none' }}
        >
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/invest"
          >
            Stocks<br />
            Advisor
          </Link>
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/advice"
          >
            Investment<br />
            Advisor
          </Link>
          <Link
            className="w-[88%] ml-[6%] mr-[6%] bg-blue-100 text-blue-950 rounded-xl pt-2 pb-2 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
            to="/budgetplaner"
          >
            Budget<br />
            Planner
          </Link>
        </div>


        <button
          className="z-50 w-25 h-25 text-3xl pb-23 font-extrabold rounded-full border-blue-950 border-4 bg-blue-800 text-blue-100 fixed bottom-5 right-5 cursor-pointer hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          onClick={AiMenuChange}
        >
          <sub>^</sub>
          <br />
          <b>AI</b>
        </button>

        <img
          className="-mt-28 w-full border-b-8 border-b-blue-900"
          src="/mainpic-above.png"
          alt="Main Visual"
        />

        <div className="flex justify-center items-center pb-5 pt-5 w-full flex-col bg-gradient-to-b from-blue-100 to-gray-400 border-b-8 border-blue-900">
          <img className="w-[30%]" src="/SmartBudget.png" alt="SmartBudget Logo" />
          <Link
            to="/faq"
            className="absolute top-[39%] right-4 bg-blue-900 pt-2 pb-2 pr-5 pl-5 rounded-2xl text-2xl font-extrabold text-blue-100 hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          >
            ?
          </Link>

          <h1 className="text-4xl font-semibold text-blue-900">
            Your Ultimate Financial Knowledge Hub
          </h1>
          <Link
            to="/learnmore"
            className="bg-blue-800 mt-10 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-2xl text-blue-100 font-bold hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
          >
            Learn More
          </Link>
        </div>

        <img className="w-full" src="/mainpic-bottom.png" alt="Footer Visual" />
      </div>
      <div className="font-sans">
        {/* Header */}
        <header className="bg-blue-900 text-white text-center py-6 mb-10">
          <h1 className="text-3xl font-bold">Top Investment Companies</h1>
          <p className="text-lg mt-2">
            Discover the most successful investment firms in the world
          </p>
        </header>

        {/* Carousel */}
        <div className="relative w-[80%] max-w-[800px] mx-auto rounded-xl overflow-hidden h-[500px] mb-10">
          <div
            className="flex transition-transform duration-500"
            style={{ transform: `translateX(-${current * 100}%)` }}
          >
            {slides.map((slide, i) => (
              <div key={i} className="min-w-full flex flex-col">
                <div className="h-[60%]">
                  <img
                    src={slide.img}
                    alt={`Slide ${i + 1}`}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="h-[20%] bg-blue-950 text-blue-100 flex items-center justify-center text-center p-4 rounded-b-xl">
                  {slide.caption}
                </div>
              </div>
            ))}
          </div>
          <button
            onClick={() => changeSlide(-1)}
            className="absolute rounded top-1/2 left-4 transform -translate-y-1/2 bg-blue-950 text-blue-100 text-3xl px-3 py-2 z-10 hover:scale-115 transition-transform hover:duration-200"
          >
            &#10094;
          </button>
          <button
            onClick={() => changeSlide(1)}
            className="absolute rounded top-1/2 right-4 transform -translate-y-1/2 bg-blue-950 text-blue-100 text-3xl px-3 py-2 z-10 hover:scale-115 transition-transform hover:duration-200"
          >
            &#10095;
          </button>
        </div>

        {/* Company Grid */}
        <div className="max-w-[1100px] mx-auto px-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-10">
          {companies.map((company, index) => (
            <div
              key={index}
              className="bg-blue-950 text-blue-100 p-6 rounded-lg shadow text-center hover:scale-105 transition-transform hover:duration-200"
            >
              <h2 className="text-xl font-semibold">
                {company.name}
              </h2>
              <p className="my-4">{company.desc}</p>
              <a
                href={company.link}
                target="_blank"
                rel="noopener noreferrer"
                className="text-blue-950 inline-block px-4 py-2 bg-blue-100 rounded hover:scale-110 transition-transform hover:duration-200 active:scale-85 active:duration-50"
              >
                Learn More
              </a>
            </div>
          ))}
        </div>
      </div>
      <Footer />
    </>
  );
}

export default Home;