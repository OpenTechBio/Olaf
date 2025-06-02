import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, firstValueFrom } from 'rxjs';
import { ChatMessage } from '../models/chat-message';
import { environment } from '../../environments/environment'

@Injectable({
  providedIn: 'root',
})
export class ChatService {
  private chatAPIEndpoint =
    environment.chatAPIEndpoint;
  private l3chatAPIEndpoint =
    environment.l3chatAPIEndpoint;
  private nameMakerAPIEndpoint =  environment.nameMakerAPIEndpoint;

  //add claude support
  constructor(private http: HttpClient) {}

  sendMessage(
    message: string,
    sessionId: string,
    userId: string,
    projectId: string,
    agent: string | null = null,
    provider: string | null = null
  ): Observable<any> {
    // Build the URL with query parameters
    let url = new URL(this.chatAPIEndpoint);
    if (agent == "L3-Reasoning"){
      url = new URL(this.l3chatAPIEndpoint);
    } 
    url.searchParams.append('user_id', userId);
    url.searchParams.append('project_id', projectId);
    if (sessionId) {
      url.searchParams.append('session_id', sessionId);
    }
    url.searchParams.append('message', message);
    if (provider == "Claude") {
      url.searchParams.append('provider', provider);
    }

    return new Observable((subscriber) => {
      const eventSource = new EventSource(url.toString());

      eventSource.onmessage = (event) => {
        subscriber.next(event.data);
      };

      eventSource.onerror = (error) => {
        subscriber.error(error);
        eventSource.close();
      };

      return () => {
        eventSource.close();
      };
    });
  }

  async generateChatNameFromHistory(history: ChatMessage[]): Promise<string> {
    // remove all images from history
    history = history.filter((message) => message.type !== 'image');
    const headers = new HttpHeaders({
      'Content-Type': 'application/json',
    });

    const response = await firstValueFrom(
      this.http.post<{ message: string }>(
        this.nameMakerAPIEndpoint,
        { history },
        {
          headers,
          responseType: 'json',
        }
      )
    );

    // Return the message from the response
    return response.message;
  }
}
