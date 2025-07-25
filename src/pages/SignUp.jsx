import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'

const SignUp = () => {
    const [fullName, setFullName] = useState('');
    const [email, setEmail] = useState('');
    const [phone, setPhone] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();

    const handleSignup = async (e) => {
        e.preventDefault();
        setError('');

        // Validate passwords match
        if (password !== confirmPassword) {
            setError('Passwords do not match');
            return;
        }

        setIsLoading(true);

        try {
            const response = await axios.post('http://localhost:5000/api/users', {
                name: fullName,
                email,
                password,
                phone
            });

            // Store token in localStorage
            if (response.data.token) {
                localStorage.setItem('userToken', response.data.token);
                localStorage.setItem('userData', JSON.stringify(response.data));
                navigate('/home');
            }
        } catch (err) {
            setError(err.response?.data?.message || 'Registration failed. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className='relative bg-no-repeat bg-cover'>
            <div className='absolute bg-[#DEDDEA] inset-0 flex justify-center items-center h-screen'>
                <div className='flex flex-col w-[50vh] px-6 py-6  bg-white z-10 rounded-xl shadow-[0_4px_6px_-1px_rgba(0,0,0,0.1),_6px_0_6px_-3px_rgba(0,0,0,0.1),_-6px_0_6px_-3px_rgba(0,0,0,0.1)] ' style={{height: "480px"}}>
                   
                        <div className='mt-1 mb-2'>
                            <h1 className='text-2xl font-bold'>Create an account !</h1>
                            <p className='text-gray-600 text-xs mt-0.5 mb-2' style={{fontSize: "13px", fontWeight: 600}}>Enter your Details</p>
                        </div>

                        {error && (
                            <div className='mb-2 p-2 bg-red-100 border border-red-400 text-red-700 rounded text-sm'>
                                {error}
                            </div>
                        )}
                        
                        <form onSubmit={handleSignup} className='flex flex-col space-y-2'>
                            <div>
                                <label className='text-gray-700 text-medium font-medium'>Full Name</label>
                                <div className='relative '>
                                    <input
                                        type="text"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm'
                                        value={fullName}
                                        onChange={(e) => setFullName(e.target.value)}
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <label className='text-gray-700 text-medium font-medium'>Email</label>
                                <div className='relative '>
                                    <input
                                        type="email"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm'
                                        value={email}
                                        onChange={(e) => setEmail(e.target.value)}
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <label className='text-gray-700 text-medium font-medium'>Phone No.</label>
                                <div className='relative'>
                                    <input
                                        type="tel"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm'
                                        value={phone}
                                        onChange={(e) => setPhone(e.target.value)}
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5a2 2 0 012-2h3.28a1 1 0 01.948.684l1.498 4.493a1 1 0 01-.502 1.21l-2.257 1.13a11.042 11.042 0 005.516 5.516l1.13-2.257a1 1 0 011.21-.502l4.493 1.498a1 1 0 01.684.949V19a2 2 0 01-2 2h-1C9.716 21 3 14.284 3 6V5z" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                            
                            <div>
                                <label className='text-gray-700 text-medium font-medium'>Password</label>
                                <div className='relative mt-0.5'>
                                    <input
                                        type="password"
                                        className='w-full py-1.5 px-2 border border-gray-300 rounded-md text-sm'
                                        value={password}
                                        onChange={(e) => setPassword(e.target.value)}
                                        required
                                    />
                                    <div className='absolute right-2 top-1.5 text-gray-500'>
                                        <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                        </svg>
                                    </div>
                                </div>
                            </div>
                            
                            <div className='pt-2'>
                                <button 
                                    type="submit" 
                                    disabled={isLoading}
                                    className='w-full bg-[#6c4298] hover:bg-[#482D64] text-white text-medium font-medium py-1.5 rounded-full'
                                >
                                    {isLoading ? 'Creating Account...' : 'Sign Up'}
                                </button>
                            </div>
                        </form>
                    </div>
                </div>
           
        </div>
    )
}

export default SignUp;