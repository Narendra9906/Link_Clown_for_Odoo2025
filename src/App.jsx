import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Login, Home, } from './pages'
import SignUp from './pages/SignUp'

const App = () => {
  const routes =(
    <Routes>
      <Route path="/home" element={<Home/>} />
      <Route path="/login" element={<Login/>} />
      <Route path="/sign-up" element={<SignUp/>} />
    </Routes>
  )
  return (
    <>
      {routes}
    </>
  )
}

export default App
