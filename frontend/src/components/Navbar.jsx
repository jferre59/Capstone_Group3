import { useNavigate } from 'react-router-dom';

export default function Navbar() {
  const navigate = useNavigate();
  return (
    <nav style={{
      display: 'flex',
      gap: '10px',
      padding: '12px 20px',
      background: '#1e293b',
      position: 'sticky',
      top: 0,
      zIndex: 9999,
      width: '100%',
      boxSizing: 'border-box',
    }}>
      <button
        onClick={() => navigate('/doctor/dashboard')}
        style={{
          color: 'white',
          background: 'none',
          border: '1px solid white',
          padding: '6px 12px',
          cursor: 'pointer',
          borderRadius: '6px',
        }}
      >
        Doctor Dashboard
      </button>
      <button
        onClick={() => navigate('/patient/input')}
        style={{
          color: 'white',
          background: 'none',
          border: '1px solid white',
          padding: '6px 12px',
          cursor: 'pointer',
          borderRadius: '6px',
        }}
      >
        Patient Input
      </button>
    </nav>
  );
}