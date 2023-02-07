import React, { Component, Fragment, Suspense} from "react";
import { MuiThemeProvider, CssBaseline, Button } from "@material-ui/core";
import { render } from "react-dom";

import FirstPage from "./FirstPage";
import MainPage from "./MainPage";
import NextPage from "./NextPage";
import FinalPage from "./FinalPage";
import PatientDataPage from "./PatientDataPage";

import theme from "../styles/theme";
import GlobalStyles from "../styles/GlobalStyles";
import Container from '@material-ui/core/Container';
import CustomScroller from 'react-custom-scroller';

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link,
  Redirect,
} from "react-router-dom";

export default class HomePage extends Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <Router>
      <MuiThemeProvider theme={theme}>
        <CssBaseline />
        <GlobalStyles />
        <Suspense fallback={<Fragment />}>
          <CustomScroller style={{ width: '100%', height: '100%' }}>
            <Container component="main" maxWidth="xl" style={{maxHeight: "95vh", overflow: 'auto'}}>
              <Switch>
                <Route exact path="/" component={FirstPage}/>
                <Route path="/create" component={MainPage} />
                <Route path="/next" component={NextPage} />
                <Route path="/final" component={FinalPage} />
                <Route path="/patientData" component={PatientDataPage} />
              </Switch>
            </Container>
          </CustomScroller>
        </Suspense>
      </MuiThemeProvider>
    </Router>
    );
  }
}