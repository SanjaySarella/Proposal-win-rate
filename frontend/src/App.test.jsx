import React,{act} from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
global.IS_REACT_ACT_ENVIRONMENT=true;
let container,root;
beforeEach(()=>{container=document.createElement('div');document.body.appendChild(container);root=createRoot(container);});
afterEach(()=>{act(()=>root.unmount());container.remove();jest.restoreAllMocks();});
const meta={records:969,agencies:['NASA'],types:['DEFINITIVE CONTRACT'],scope:'Historical sample',sha256:'test'};
test('shows actual research response and empty-selection boundary',async()=>{
 global.fetch=jest.fn().mockResolvedValueOnce({ok:true,json:async()=>meta}).mockResolvedValueOnce({ok:true,json:async()=>({count:0,total:0,median:null,records:[],method:'Snapshot only'})});
 await act(async()=>root.render(<App/>));
 await act(async()=>container.querySelector('form').dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
 expect(container.textContent).toContain('No awards match these filters');
 expect(container.textContent).not.toContain('WIN PROBABILITY');
 expect(JSON.parse(global.fetch.mock.calls[1][1].body).agency).toBe(null);
});
test('renders network failure instead of fabricated results',async()=>{
 global.fetch=jest.fn().mockRejectedValue(new Error('Unavailable'));
 await act(async()=>root.render(<App/>));
 expect(container.querySelector('[role="alert"]').textContent).toContain('Unavailable');
 expect(container.querySelector('button').disabled).toBe(true);
});
