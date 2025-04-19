import { useState} from 'react';
import './styles/index.css';

function TopCompanies() {

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
        <div className='flex items-center flex-col mt-10'>
            {/* Header */}
            <header className="bg-blue-950 text-white text-center py-6 mb-10 max-w-max rounded-xl p-6">
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
                    className="absolute scale-150 rounded top-1/2 left-8 transform -translate-y-1/2 bg-blue-100 text-blue-950 text-3xl px-3 py-1 z-10 hover:scale-175 transition-transform hover:duration-200 active:duration-50 active:scale-130"
                >
                    &#10094;
                </button>
                <button
                    onClick={() => changeSlide(1)}
                    className="absolute scale-150 rounded top-1/2 right-8 transform -translate-y-1/2 bg-blue-100 text-blue-950 text-3xl px-3 py-1 z-10 hover:scale-175 transition-transform hover:duration-200 active:duration-50 active:scale-130"
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

    );
}

export default TopCompanies;