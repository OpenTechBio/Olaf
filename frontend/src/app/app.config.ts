import { provideHttpClient, withFetch } from '@angular/common/http';
import { ApplicationConfig, provideZoneChangeDetection } from '@angular/core';
import { provideRouter } from '@angular/router';
import { provideClientHydration } from '@angular/platform-browser';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';

import { initializeApp, provideFirebaseApp } from '@angular/fire/app';
import { getAuth, provideAuth } from '@angular/fire/auth';
import { getFirestore, provideFirestore } from '@angular/fire/firestore';
import { getFunctions, provideFunctions } from '@angular/fire/functions';
import { getStorage, provideStorage } from '@angular/fire/storage';

import { provideHighlightOptions } from 'ngx-highlightjs';

import { routes } from './app.routes';

export const appConfig: ApplicationConfig = {
  providers: [
    provideHttpClient(withFetch()),
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideRouter(routes),
    provideClientHydration(),
    provideAnimationsAsync(),

    provideFirebaseApp(() =>
      initializeApp({
        projectId: 'twocube-web',
        appId: '1:REMOVED:web:349739f3904ebe5c7a28ef',
        storageBucket: 'twocube-web.appspot.com',
        apiKey: 'REMOVED',
        authDomain: 'twocube-web.firebaseapp.com',
        messagingSenderId: 'REMOVED',
        measurementId: 'REMOVED',
      }),
    ),
    provideAuth(() => getAuth()),
    provideFirestore(() => getFirestore()),
    provideFunctions(() => getFunctions()),
    provideStorage(() => getStorage()),

    provideHighlightOptions({
      fullLibraryLoader: () => import('highlight.js'),
    }),
  ],
};
