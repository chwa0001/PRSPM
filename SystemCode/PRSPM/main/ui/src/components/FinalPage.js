import React ,{useState,useEffect,useRef} from 'react';
import Avatar from '@material-ui/core/Avatar';
import CssBaseline from '@material-ui/core/CssBaseline';
import Grid from '@material-ui/core/Grid';
import LockOutlinedIcon from '@material-ui/icons/LockOutlined';
import Typography from '@material-ui/core/Typography';
import { makeStyles,useTheme,styled} from '@material-ui/core/styles';
import Cookies, { set } from 'js-cookie';
import Fade from '@material-ui/core/Fade';
import CircularProgress from '@material-ui/core/CircularProgress';
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import Chip from '@material-ui/core/Chip';
import Slider from '@material-ui/core/Slider';
import Divider from '@material-ui/core/Divider';

const useStyles = makeStyles((theme) => ({
  paper: {
    marginTop: theme.spacing(0),
    display: 'flex',
    flexDirection: 'column',
    width:'100%',
    alignItems:'stretch',
  },
  header: {
    marginTop: theme.spacing(0),
    display: 'flex',
    flexDirection: 'column',
    width:'100%',
    alignItems:'center',
    marginBottom: theme.spacing(5),
  },
  avatar: {
    margin: theme.spacing(1),
    backgroundColor: theme.palette.secondary.main,
  },
  form: {
    width: '100%', // Fix IE 11 issue.
    marginTop: theme.spacing(0),
  },
  submit: {
    backgroundColor:"#0a57f2",
    color: 'white',
    height: 36,
    margin: theme.spacing(3, 0, 2),
  },
  formControl: {
    display: 'flex',
    margin: theme.spacing(1),
    minWidth: 25,
  },
  backbutton: {
    backgroundColor: '#f20a40',
    color: 'white',
    height: 36,
  },
}));

const PrettoSlider = styled(Slider)({
  color: '#52af77',
  height: 8,
  flexGrow:1,
  '& .MuiSlider-track': {
    border: 'none',
    height:5,
  },
  '& .MuiSlider-thumb': {
    height: 15,
    width: 15,
    backgroundColor: '#fff',
    border: '2px solid currentColor',
    '&:focus, &:hover, &.Mui-active, &.Mui-focusVisible': {
      boxShadow: 'inherit',
    },
    '&:before': {
      display: 'none',
    },
  },
  '& .MuiSlider-valueLabel': {
    lineHeight: 1.2,
    fontSize: 12,
    background: 'unset',
    padding: 0,
    width: 32,
    height: 32,
    borderRadius: '50% 50% 50% 50%',
    backgroundColor: '#52af77',
    
  },
});

export default function FinalPage() {
  const classes = useStyles();
  const [loading,setLoading] = useState(true);
  const [reply, getReply] = useState([]);
  const [hospitaliseRate,setHospitaliseRate] = useState(0);
  const [threshold,setThreshold] = useState(0);
  const [sliderValue,setSliderValue] = useState(0);
  const [isRunning,setIsRunning]=useState(0);
  const intervalIdRef = useRef(0);

  const patientID = Cookies.get('id');
  async function GetPrediction(patientID) {
    const response = await fetch(`/HospitalisedPrediction?id=${patientID}`);
    if (!response.ok) {
      const message = `An error has occured: ${response.status}`;
      throw new Error(message);
    }
    else{
      const data = await response.json();
      setThreshold(data.threshold*100);
      setHospitaliseRate(data.predictionRate*100);
      setIsRunning(1);
    }
  }
  useEffect(() => {
    GetPrediction(patientID,'').catch(console.error);
  }, []);

  async function GetArticles(patientID) {
    const response = await fetch(`/ArticlePrediction?id=${patientID}`);
    if (!response.ok) {
      const message = `An error has occured: ${response.status}`;
      throw new Error(message);
    }
    else{
      const data = await response.json();
      setLoading(false);
      let articleList = []
      for (let i = 0; i < Object.keys(data.ArticlePredicted).length; i++) {
        articleList.push(data.ArticlePredicted[Object.keys(data.ArticlePredicted)[i]])
      }
      getReply(articleList);
      console.log(articleList)
    }
  }
  useEffect(() => {
    GetArticles(patientID,'').catch(console.error);
  }, []);


  useEffect(() => {
    if(isRunning){
      intervalIdRef.current = setInterval(() => {
        setSliderValue((v) =>{
          return v + 0.5
        });
    }, 20);
    }
    return () =>clearInterval(intervalIdRef.current)
  }, [isRunning]);

  useEffect(() => {
    if(sliderValue>hospitaliseRate){
      setIsRunning(0)
    }
  }, [sliderValue]);

  return (
    <Card style={{maxWidth:'80%',margin:'auto'}}>
      <CssBaseline />
      <div className={classes.header}>
      <Avatar className={classes.avatar}>
          <LockOutlinedIcon />
        </Avatar>
      <Typography variant="h5" style={{fontSize:'20px',marginLeft:'20px'}}>
          {loading?"We are preparing the prediction outcome from the model...":"Adverse event evaluation outcome"}
      </Typography>
      <Fade in={loading} unmountOnExit>
        <Grid style={{marginTop:'10px'}}>
          <CircularProgress
            size={24}
            sx={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              marginTop: '-12px',
              marginLeft: '-12px',
            }}
          />
        </Grid>
      </Fade>
      </div>
      <div className={classes.paper}>
        
        <CardContent sx={{ width: 1 }}>
        <Grid container direction="row" alignItems="stretch">
        <Fade in={!loading && hospitaliseRate!=0 } unmountOnExit>
        <Grid item xs={12}>
        <PrettoSlider
        valueLabelDisplay="on"
        aria-label="pretto slider"
        defaultValue={20}
        valueLabelFormat={(value)=>(value<hospitaliseRate?<div>{value.toFixed(2)}</div>:<div>Your hospitalisation Rate is {hospitaliseRate.toFixed(2)}%</div>)}
        max={100}
        min={0}
        value={sliderValue}
        style={hospitaliseRate>threshold?{color:'orange'}:{}}
        />
        <Fade in={sliderValue>=hospitaliseRate-1 && hospitaliseRate>threshold && hospitaliseRate!=0} unmountOnExit>
        <Typography variant="body1" style={{marginBottom:'5px',color:'red'}}>
            Please call +65 6321 4311 for emergency care under Singapore General Hospital! 
        </Typography>
        </Fade>
        <Fade in={sliderValue>=hospitaliseRate-1 && hospitaliseRate<=threshold && hospitaliseRate!=0} unmountOnExit>
        <Typography variant="body1" style={{marginBottom:'5px',color:'green'}}>
          You do not have high risk. Please continue to monitor your symptoms at home, and report again if it worsens.
        </Typography>
        </Fade>
        </Grid>
        </Fade>
        </Grid>
        </CardContent>
        <Fade in={!loading} unmountOnExit>
        <CardContent>
        <Grid container direction="row" spacing={3}>
        <Grid item xs={12}>
        <Typography variant="body1" style={{marginBottom:'10px'}}>
        Here are some recommended medical articles related to your reported adverse event, for your reading and information:
        </Typography>
        </Grid>     
       
       {reply.map((value)=>{return(
        <Grid item xs={12} key={value.name}>
          <Typography variant="body1" style={{marginBottom:'20px'}}>
            {value.text}
          </Typography>
          <Grid item xs={12} container direction="row" spacing={2}>
          <Grid item xs={3}>
          <Typography variant="body2">
            Title:
          </Typography>
          </Grid>  
          <Grid item xs>
          <Typography variant="body2">
            {value.title}
          </Typography>
          </Grid>
          </Grid>
          <Grid item xs={12} container direction="row" spacing={2}>
          <Grid item xs={3}>
          <Typography variant="body2">
            Authors:
          </Typography>
          </Grid>  
          <Grid item xs>
          <Typography variant="body2">
            {value.authors}
          </Typography>
          </Grid>
          </Grid>
          <Grid item xs={12} container direction="row" spacing={2}>
          <Grid item xs={3}>
          <Typography variant="body2">
          Journal:
          </Typography>
          </Grid>  
          <Grid item xs>
          <Typography variant="body2">
            {value.journal}
          </Typography>
          </Grid>
          </Grid>
          <Grid item xs={12} container direction="row" spacing={2}>
          <Grid item xs={3}>
          <Typography variant="body2">
          Publish date:
          </Typography>
          </Grid>  
          <Grid item xs>
          <Typography variant="body2">
            {value.publish_time}
          </Typography>
          </Grid>
          </Grid>
          <Grid item xs={12} container direction="row" spacing={2}>
          <Grid item xs={3}>
          <Typography variant="body2">
          Alert:
          </Typography>
          </Grid>  
          <Grid item xs>
          <Chip style={value.sentiment.neg>0.5?{backgroundColor:'orange',marginBottom:'5px'}:{backgroundColor:'grey',marginBottom:'5px'}}/>
          </Grid>
          </Grid>
          <Divider/>
        </Grid>
       )})}
        </Grid>
        </CardContent>
        </Fade>
      </div>
    </Card>
  );
}