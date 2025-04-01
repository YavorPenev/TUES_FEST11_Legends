import { useState, useRef, useEffect } from 'react';
import { Link } from 'react-router';

function Home() {
  const topmenuRef = useRef(null);
  const [AiStatus, SetAiStatus] = useState(0);

  useEffect(() => {
    if (topmenuRef.current) {
      topmenuRef.current.style.display = AiStatus === 1 ? "flex" : "none";
    }
  }, [AiStatus]); // Runs every time AiStatus changes

  const AiMenuChange = () => {
    SetAiStatus((prevStatus) => (prevStatus + 1) % 2);
  };

  return (
    <>
      <header className='pl-5 pr-5 fixed top-0 left-0 w-full bg-blue-800 border-blue-900 flex justify-between p-4 border-b-4 bg-gradient-to-b from-gray-900 to-transparent items-center'>
        <Link className='bg-blue-100 pr-[1%] pl-[1%] pt-[0.4%] pb-[0.4%] rounded-xl hover:scale-110 transition-transform duration-200' to="/learnmore">
          <img title='SmartBudget' alt='logo' src='/logo2.png' className='max-h-12 aspect-auto' />
        </Link>
        <h1 className='text-white font-bold text-3xl ml-4'>SmartBudget</h1>
        <Link className='bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200' to="/articles">Articles</Link>
        <Link className='bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200' to="/login">Login</Link>
        <Link className='bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200' to="/signup">Sign Up</Link>
        <Link className='bg-blue-100 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-xl text-blue-950 font-bold hover:scale-110 transition-transform duration-200' to="/account">My Account</Link>
        <Link className='bg-blue-100 pt-2 pb-2 pr-5 pl-5 rounded-2xl text-2xl font-extrabold text-blue-950 hover:scale-110 transition-transform duration-200' to="/faq">?</Link>
      </header>

      {/* AI Menu */}
      <div ref={topmenuRef} className='flex-col-reverse rounded-t-full fixed right-5 bottom-15 pb-12 pt-5 gap-4 text-blue-100 w-20 bg-blue-800 border-4 border-blue-950' style={{ display: "none" }}>
        <Link className='w-[80%] bg-blue-100 text-blue-950 rounded-xl padding-2 mr-[10%] ml-[10%] hover:scale-110 transition-transform duration-200' to="/stockai">Stock<br />Helper</Link>
        <Link className='w-[80%] bg-blue-100 text-blue-950 rounded-xl padding-2 mr-[10%] ml-[10%] hover:scale-110 transition-transform duration-200' to="/budgetai">Budget<br />Planner</Link>
      </div>

      {/* AI Button */}
      <button
        className='w-20 h-20 pb-10 text-2xl rounded-full border-blue-950 border-4 bg-blue-800 text-blue-100 fixed bottom-5 right-5 cursor-pointer hover:scale-110 transition-transform duration-200'
        onClick={AiMenuChange}
      >
        <sub>^</sub><br /><b>AI</b>
      </button>

      <img className='mt-2 w-full border-b-8 border-b-blue-900' src='/mainpic-above.png' />

      <div className='flex justify-center items-center pb-5 pt-5 w-full flex-col bg-gradient-to-b from-blue-100 to-gray-400 border-b-8 border-blue-900'>
        <img className='w-[30%]' src='/SmartBudget.png' />
        <h1 className='text-4xl font-semibold text-blue-900'>Your Ultimate Financial Knowledge Hub</h1>
        <Link className='bg-blue-800 mt-10 pl-3 pr-3 pt-2 pb-2 rounded-2xl text-2xl text-blue-100 font-bold hover:scale-110 transition-transform duration-200' to="/learnmore">Learn More</Link>
      </div>

      <img className='w-full' src='/mainpic-bottom.png' />
    </>
  );
}

export default Home;
