/*import { useState, useEffect } from 'react'
//import reactLogo from './assets/react.svg'
//import viteLogo from '/vite.svg'
import './App.css'
import axios from 'axios'

function App() {
  const [count, setCount] = useState(0)

  /////////////////////////////////////////////////////////

  const [array, setArray] = useState([]);// test connection

  const fetchAPI = async () => {// connect to backend
    const response = await axios.get("http://localhost:8000/api");
    setArray(response.data.fruit);
    console.log(response.data.fruit);
  }

  useEffect(() => {// izwikwane na funkciqta
    fetchAPI();
  }, []);

  /////////////////////////////////////////////////////////

  const [currentSlide, setCurrentSlide] = useState(0);
  const slides = [
    { image: '/slide1.jpg', text: 'This is Slide 1' },
    { image: '/slide2.jpg', text: 'This is Slide 2' },
    { image: '/slide3.jpg', text: 'This is Slide 3' },
    { image: '/slide4.jpg', text: 'This is Slide 4' },
    { image: '/slide5.jpg', text: 'This is Slide 5' },
  ];

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prevSlide) => (prevSlide + 1) % slides.length);
    }, 5000); 
    return () => clearInterval(interval);
  }, [slides.length]);


  const nextSlide = () => {
    setCurrentSlide((prevSlide) => (prevSlide + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prevSlide) => (prevSlide - 1 + slides.length) % slides.length);
  };

    /////////////////////////////////////////////////////////

  return (
    <>
      <h1 className='text-green-500 title' ><b>Welcome to MyNote</b></h1>

      <div className='header'>
        <button className='Hbutton'><b>New note</b></button>
        <button className='Hbutton'><b>Redact note</b></button>
        <button className='Hbutton'><b>Delete note</b></button>
        <button className='Hbutton'><b>View notes</b></button>
      </div>


      <div className="carousel">
        {slides.map((slide, index) => (
          <div
            key={index}
            className={`carousel-slide ${index === currentSlide ? 'active' : ''}`}
          >
            <img src={slide.image} alt={`Slide ${index + 1}`} className="carousel-image" />
            <p className="carousel-text">{slide.text}</p>
          </div>
        ))}
        <button className="carousel-button prev" onClick={prevSlide}>❮</button>
        <button className="carousel-button next" onClick={nextSlide}>❯</button>
      </div>

      <button onClick={() => setCount((count) => count + 1)}>
        count is {count}
      </button>

      {
        array.map((element, index) =>
          <div key={index}>
            <p><b> {element};</b></p>
          </div>
        )
      }
    </>
  )
}

export default App*/


import { useState, useEffect } from 'react';
import { Routes, Route, useNavigate } from 'react-router-dom';
import './styles/App.css';
import View from './View'; // Импортирайте View.jsx
import Add from './Add';
import Del from './delete';
import Redact from './redact';
import { fetchAPI }  from './network/index'


function App() {
  const [count, setCount] = useState(0);
  const [array, setArray] = useState([]); // Test connection
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    { image: '/slide1.jpg', text: 'This is Slide 1' },
    { image: '/slide2.jpg', text: 'This is Slide 2' },
    { image: '/slide3.jpg', text: 'This is Slide 3' },
    { image: '/slide4.jpg', text: 'This is Slide 4' },
    { image: '/slide5.jpg', text: 'This is Slide 5' },
  ];

  
  useEffect(() => {
    fetchAPI(setArray);
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentSlide((prevSlide) => (prevSlide + 1) % slides.length);
    }, 5000);
    return () => clearInterval(interval);
  }, [slides.length]);

  const nextSlide = () => {
    setCurrentSlide((prevSlide) => (prevSlide + 1) % slides.length);
  };

  const prevSlide = () => {
    setCurrentSlide((prevSlide) => (prevSlide - 1 + slides.length) % slides.length);
  };

  const navigate = useNavigate(); // React Router hook за навигация

  return (
    <Routes>
      {/* Главна страница */}
      <Route
        path="/"
        element={
          <>
            <h1 className="text-green-500 title"><b>Welcome to MyNote</b></h1>

            <div className="header">
              <button className="Hbutton" onClick={() => navigate('/add')}><b>New note</b></button>
              <button className="Hbutton" onClick={() => navigate('/redact')}><b>Redact note</b></button>
              <button className="Hbutton" onClick={() => navigate('/delete')}><b>Delete note</b></button>
              <button className="Hbutton" onClick={() => navigate('/notes')}><b>View notes</b></button>
            </div>

            <div className="carousel">
              {slides.map((slide, index) => (
                <div
                  key={index}
                  className={`carousel-slide ${index === currentSlide ? 'active' : ''}`}
                >
                  <img src={slide.image} alt={`Slide ${index + 1}`} className="carousel-image" />
                  <p className="carousel-text">{slide.text}</p>
                </div>
              ))}
              <button className="carousel-button prev" onClick={prevSlide}>❮</button>
              <button className="carousel-button next" onClick={nextSlide}>❯</button>
            </div>

            <button onClick={() => setCount((count) => count + 1)}>
              count is {count}
            </button>

            {
        array.map((element, index) =>
          <div key={index}>
            <p><b> {element};</b></p>
          </div>
        )
      }
          </>
        }
      />

      {/* Страница за преглед на бележки */}
      <Route path="/notes" element={<View />} />
      <Route path="/add" element={<Add />} />
      <Route path="/delete" element={<Del />} />
      <Route path="/redact" element={<Redact />} />

    </Routes>
  );
}

export default App;